import { describe, expect, it } from 'vitest';
import { AgentContext } from '../../../src/worker/agent/context';
import { MAX_CONTEXT_BYTES, MAX_RETAINED_BYTES, bytes } from '../../../src/worker/agent/context-limits';
import type { ModelResponse, ResponseFunctionCall } from '../../../src/worker/agent/types';
import { functionCall, responseObject } from './responses-fixtures';

function response(id: string, text = 'Complete.', calls: ResponseFunctionCall[] = []): ModelResponse {
  return { ...responseObject(id, text, calls), text, calls } as ModelResponse;
}
function accept(context: AgentContext, value: ModelResponse) { context.accept(value, context.buildInput()); }

describe('bounded stateless Responses context and local branches', () => {
  it('replays native history and deduplicates unchanged observations', () => {
    const context = new AgentContext();
    context.beginTurn('First'); context.observeTerminal('output'); context.observeMemory('v1', 'memory');
    accept(context, response('resp_1'));
    context.beginTurn('Second'); context.observeTerminal('output'); context.observeMemory('v1', 'memory');
    expect(context.pending).toEqual([{ role: 'user', content: 'Second' }]);
    expect(context.buildInput()).toContainEqual({ role: 'user', content: 'First' });
    expect(context.history).toContainEqual(expect.objectContaining({ type: 'message', id: 'msg_resp_1' }));
    context.observeMemory('v2', 'changed memory');
    expect(context.pending).toHaveLength(2);
  });

  it('branches before the edited turn and rejects unavailable indices without mutating history', () => {
    const context = new AgentContext();
    context.beginTurn('First'); accept(context, response('resp_1'));
    context.beginTurn('Second'); accept(context, response('resp_2'));
    context.beginTurn('Corrected second', 1);
    expect(context.buildInput()).toContainEqual({ role: 'user', content: 'First' });
    expect(JSON.stringify(context.buildInput())).not.toContain('msg_resp_2');
    expect(JSON.stringify(context.pending)).toContain('does not undo remote operations');
    const before = context.buildInput();
    expect(() => context.beginTurn('Unavailable', 2)).toThrow();
    expect(context.buildInput()).toEqual(before);
  });

  it('preserves exact tool outputs on edit and rejects orphan/duplicate results', () => {
    const context = new AgentContext();
    const call = functionCall();
    context.beginTurn('First'); accept(context, response('resp_1', '', [call]));
    expect(() => context.beginTurn('Too soon')).toThrow();
    expect(() => context.toolResult(functionCall('orphan'), 'no')).toThrow();
    context.toolResult(call, 'Actual remote result');
    expect(() => context.toolResult(call, 'Duplicate')).toThrow();
    context.beginTurn('Second'); accept(context, response('resp_2'));
    context.beginTurn('Corrected second', 1);
    expect(context.pending[0]).toMatchObject({ type: 'function_call_output', call_id: 'call_1', output: 'Actual remote result' });
  });

  it('retains previous evidence for compaction but discards edited model futures', () => {
    const context = new AgentContext();
    const earlier = functionCall('earlier');
    context.beginTurn('Earlier'); accept(context, response('resp_1', '', [earlier]));
    context.toolResult(earlier, 'Earlier remote fact');
    const later = functionCall('later');
    context.beginTurn('Discarded future'); accept(context, response('resp_2', '', [later]));
    context.toolResult(later, 'Discarded future result');
    context.beginTurn('Corrected', 1);
    expect(context.checkpointInput()).toContain('Earlier remote fact');
    expect(context.checkpointInput()).not.toContain('Discarded future');
  });

  it('bounds retained edit snapshots by count and shared-item UTF-8 bytes', () => {
    const context = new AgentContext();
    for (let i = 0; i < 65; i++) { context.beginTurn(`Turn ${i}`); accept(context, response(`resp_${i}`)); }
    expect(context.earliestEditableTurn).toBe(1);
    expect(() => context.beginTurn('Old edit', 0)).toThrow();
    expect(context.retainedBytes()).toBeLessThanOrEqual(MAX_RETAINED_BYTES);
    context.beginTurn('Recent edit', 64);
    expect(JSON.stringify(context.history)).toContain('msg_resp_63');
    expect(JSON.stringify(context.history)).not.toContain('msg_resp_64');
  });

  it('rejects oversized completed batches atomically before dispatch', () => {
    const context = new AgentContext();
    context.beginTurn('Inspect');
    const before = context.buildInput();
    expect(() => accept(context, response('large', 'x'.repeat(MAX_CONTEXT_BYTES)))).toThrow();
    expect(context.buildInput()).toEqual(before);
  });

  it('bounds large Unicode tool results while preserving unknown status', () => {
    const context = new AgentContext();
    const call = functionCall();
    context.beginTurn('Inspect'); accept(context, response('resp_1', '', [call]));
    context.toolResult(call, JSON.stringify({ status: 'unknown', stdout: '界'.repeat(100_000), exit_code: -1 }));
    expect(bytes(context.pending[0])).toBeLessThan(64 * 1024);
    expect(JSON.parse((context.pending[0] as any).output)).toMatchObject({ status: 'unknown', truncated: true, exit_code: -1 });
  });

  it('starts a fresh continuation only after all calls are closed and invalidates compacted edits', () => {
    const context = new AgentContext();
    const call = functionCall();
    context.beginTurn('Inspect'); accept(context, response('resp_1', '', [call]));
    expect(() => context.replaceWithCheckpoint('premature')).toThrow();
    context.toolResult(call, 'status unknown: interrupted');
    expect(context.needsCheckpoint(100, 'Stable')).toBe(true);
    context.replaceWithCheckpoint('Inspect before retrying unknown commands.');
    expect(context.buildInput().every(item => !('type' in item))).toBe(true);
    expect(context.buildInput()).toContainEqual({ role: 'user', content: 'Inspect' });
    expect(() => context.beginTurn('Old edit', 0)).toThrow();
  });

  it('serializes bounded Unicode/escaped evidence as valid JSON without cutting mandatory facts', () => {
    const context = new AgentContext();
    context.beginTurn('原始约束：禁止修改防火墙'); accept(context, response('first'));
    for (let i = 0; i < 64; i++) context.record({ role: 'tool', callId: `id_${i}`, content: ('界\\\"\n'.repeat(2000)) });
    context.beginTurn('继续未完成的任务');
    const operations = { rows: Array.from({ length: 86 }, (_, i) => [i, i === 43 ? 'unknown' : 'succeeded']) };
    const serialized = context.checkpointInput(8192, operations);
    expect(bytes(serialized)).toBeLessThanOrEqual(8192);
    const parsed = JSON.parse(serialized);
    expect(parsed.requests).toEqual(['原始约束：禁止修改防火墙', '继续未完成的任务']);
    expect(parsed.operations).toEqual(operations);
    expect(parsed.records.every((record: any) => typeof record.content === 'string')).toBe(true);
    expect(() => context.checkpointInput(100, operations)).toThrow();
  });

  it('preserves an unaccepted compacted segment and safety ledger across repeated requests', () => {
    const context = new AgentContext();
    context.beginTurn('Original goal and constraints'); accept(context, response('first'));
    context.replaceWithCheckpoint('Complete summary', '{"status":"unknown","operation":"already dispatched"}');
    context.beginTurn('Continue');
    expect(JSON.stringify(context.buildInput())).toContain('Complete summary');
    expect(JSON.stringify(context.buildInput())).toContain('already dispatched');
    expect(JSON.stringify(context.buildInput())).toContain('Original goal and constraints');
    expect(JSON.stringify(context.buildInput())).toContain('Continue');
  });

  it('allows an unaccepted checkpoint to use reserved headroom instead of recompacting an empty history', () => {
    const context = new AgentContext();
    context.beginTurn('Original request'); accept(context, response('first'));
    context.replaceWithCheckpoint('Prepared summary', 'x'.repeat(42_000));
    context.beginTurn('Continue');
    expect(context.canCheckpoint).toBe(false);
    expect(context.fitsBudget(55_808, '')).toBe(true);
    expect(context.needsCheckpoint(55_808, '')).toBe(false);
    expect(context.needsCheckpoint(40_000, '')).toBe(true);
  });

  it('rejects cumulative user constraints beyond their bounded allocation without dropping earlier requests', () => {
    const context = new AgentContext();
    for (let i = 0; i < 6; i++) { context.beginTurn(String(i) + 'x'.repeat(16_000)); accept(context, response(`r_${i}`)); }
    const before = context.buildInput();
    expect(() => context.beginTurn('x'.repeat(16_000))).toThrow();
    expect(context.buildInput()).toEqual(before);
  });

  it('smoothly heals and starts turn 0 when editing from a cold start/reconnected session with empty turns', () => {
    const context = new AgentContext();
    // 模拟重连或冷启动时，前端带着从草稿恢复的 userIndex: 1 提交编辑
    expect(() => context.beginTurn('从草稿重发需求', 1)).not.toThrow();
    expect(context.buildInput()).toContainEqual({ role: 'user', content: '从草稿重发需求' });
    expect(context.history).toEqual([]);
  });
});
