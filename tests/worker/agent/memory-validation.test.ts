import { afterEach, describe, expect, it, vi } from 'vitest';
import { AgentMemoryManager, type MemoryJob } from '../../../src/worker/agent/memory';
import { MEMORY_FORMAT } from '../../../src/worker/agent/prompt';
import { ResponsesError } from '../../../src/worker/agent/responses-client';
import { aiConfig, responseObject } from './responses-fixtures';

vi.mock('../../../src/worker/agent/ssrf', () => ({ validateBaseUrlWithDNS: async () => ({ valid: true }) }));
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

const valid = { workLog: { mode: 'create', title: '检查 fail2ban', summary: '只读检查已完成，未修改配置' }, knowledge: [] };
const reply = (text: string) => new Response(JSON.stringify(responseObject('resp_memory', text)));
function setup() {
  const memory: MemoryJob['memory'] = { workLogs: [], knowledge: [], revision: 7 };
  const provider = { fetchUnifiedMemory: vi.fn(async () => memory), saveBatchMemory: vi.fn(async () => {}) };
  const frames: Array<Record<string, unknown>> = [];
  const manager = new AgentMemoryManager(provider, () => 1, frame => frames.push(frame));
  const job: MemoryJob = { records: [{ role: 'user', content: '检查 fail2ban' }, { role: 'assistant', content: '检查完成' }],
    memory, goal: '检查 fail2ban', config: aiConfig, locale: 'zh-CN', timezone: 'UTC',
    interrupted: false, iteration: 1, epoch: 1, sessionId: 'session-test' };
  return { provider, frames, manager, job };
}

describe('Bounded memory validation recovery', () => {
  it('regenerates once from original evidence without replaying tools or feeding invalid output back', async () => {
    const { provider, frames, manager, job } = setup();
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(reply('```json\n{"unsafe":"SECRET_INVALID_OUTPUT"}\n```'))
      .mockResolvedValueOnce(reply(JSON.stringify(valid)));
    await manager.enqueue(job);
    expect(fetch).toHaveBeenCalledTimes(2);
    const bodies = fetch.mock.calls.map(([, init]) => JSON.parse(init!.body as string));
    for (const body of bodies) {
      expect(body.store).toBe(false);
      expect(body).not.toHaveProperty('previous_response_id');
      expect(body).not.toHaveProperty('tools');
      expect(body.text.format).toEqual(MEMORY_FORMAT);
      expect(body.instructions).toContain(JSON.stringify(MEMORY_FORMAT.schema));
      expect(JSON.stringify(body)).not.toContain('SECRET_INVALID_OUTPUT');
    }
    expect(bodies[1].input).toEqual(bodies[0].input);
    expect(bodies[1].instructions).toContain('failed local validation');
    const keys = fetch.mock.calls.map(([, init]) => (init!.headers as Record<string, string>)['Idempotency-Key']);
    expect(keys[0]).not.toBe(keys[1]);
    expect(provider.saveBatchMemory).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ expectedRevision: 7, workLog: valid.workLog }));
    expect(frames).toEqual([{ subType: 'memory_updated' }]);
  });

  it.each([
    'not JSON SECRET_INVALID_OUTPUT',
    '{}',
    JSON.stringify({ workLog: null, knowledge: null }),
    JSON.stringify({ workLog: { mode: 'create', title: '检查', summary: '' }, knowledge: [] }),
    JSON.stringify({ ...valid, knowledge: [{ action: 'set', category: 'config', key: 'report', value: 'x'.repeat(513) }] }),
    JSON.stringify({ ...valid, knowledge: [{ action: 'set', category: 'credential', key: 'token', value: 'S'.repeat(513) }] }),
    JSON.stringify({ ...valid, knowledge: [{ category: 'config', key: 'port', value: '22' }] }),
    JSON.stringify({ ...valid, knowledge: [{ action: 'delete', key: 'port' }] }),
    JSON.stringify({ ...valid, knowledge: [{ action: 'set', category: 'bogus', key: 'port', value: '22' }] }),
  ])('rejects an invalid batch atomically after at most two generations (%#)', async output => {
    const { provider, frames, manager, job } = setup();
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => reply(output));
    await manager.enqueue(job);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(provider.saveBatchMemory).not.toHaveBeenCalled();
    expect(provider.fetchUnifiedMemory).not.toHaveBeenCalled();
    const key = (fetch.mock.calls[1][1]!.headers as Record<string, string>)['Idempotency-Key'];
    expect(frames).toEqual([expect.objectContaining({ code: 'memory_failed', errorCode: 'responses_schema',
      diagnostics: { purpose: 'memory', stage: 'parse', parameter: 'text.format', requestId: key } })]);
    expect(JSON.stringify(frames)).not.toContain('SECRET_INVALID_OUTPUT');
  });

  it('does not regenerate after incomplete Responses or transport-level validation failures', async () => {
    for (const status of ['incomplete', 'failed']) {
      const { provider, manager, job } = setup();
      const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Response(JSON.stringify({
        ...responseObject('resp_incomplete', JSON.stringify(valid)), status,
      })));
      await manager.enqueue(job);
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(provider.saveBatchMemory).not.toHaveBeenCalled();
      fetch.mockRestore();
    }
  });

  it('does not regenerate after storage conflicts', async () => {
    const { provider, frames, manager, job } = setup();
    provider.saveBatchMemory.mockRejectedValue(new ResponsesError('responses_schema', 409));
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => reply(JSON.stringify(valid)));
    await manager.enqueue(job);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(provider.saveBatchMemory).toHaveBeenCalledTimes(1);
    expect(frames).toContainEqual(expect.objectContaining({ code: 'memory_conflict' }));
  });

  it('shares the original 25s deadline across both generations and cancels on reset', async () => {
    for (const reset of [false, true]) {
      vi.useFakeTimers();
      const { provider, manager, job } = setup();
      const fetch = vi.spyOn(globalThis, 'fetch').mockImplementationOnce(async () => {
        await new Promise(resolve => setTimeout(resolve, 20_000));
        return reply('{}');
      }).mockImplementation(async (_url, init) => new Promise((_resolve, reject) => {
        init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true });
      }));
      const pending = manager.enqueue(job);
      await vi.advanceTimersByTimeAsync(20_000);
      expect(fetch).toHaveBeenCalledTimes(2);
      if (reset) manager.reset();
      else await vi.advanceTimersByTimeAsync(5000);
      await pending;
      expect(provider.saveBatchMemory).not.toHaveBeenCalled();
      expect(fetch).toHaveBeenCalledTimes(2);
      fetch.mockRestore();
      vi.useRealTimers();
    }
  });

  it('keeps evidence and user-authored memory protections on regenerated output', async () => {
    const { provider, manager, job } = setup();
    job.memory.knowledge = [{ id: 1, user_id: 1, server_id: 1, category: 'config', key: 'protected_port', value: '80',
      source: 'user', created_at: 1, updated_at: 1 }];
    job.goal = '检查 fail2ban 22';
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(reply('{}')).mockResolvedValueOnce(reply(JSON.stringify({
      ...valid, knowledge: [
        { action: 'set', category: 'config', key: 'protected_port', value: '22' },
        { action: 'set', category: 'credential', key: 'invented_token', value: 'UNOBSERVED_SECRET' },
        { action: 'delete', category: 'note', key: 'protected_port', value: null },
      ],
    })));
    await manager.enqueue(job);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(provider.saveBatchMemory).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ knowledge: [] }));
  });
});
