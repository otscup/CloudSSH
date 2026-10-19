import { afterEach, describe, expect, it, vi } from 'vitest';
import { AgentCore } from '../../src/worker/agent/core';
import { TerminalContext } from '../../src/worker/agent/terminal-context';
import { aiConfig, checkpoint, functionCall, reply, responseObject, sse } from './agent/responses-fixtures';

vi.mock('../../src/worker/agent/ssrf', () => ({ validateBaseUrlWithDNS: async () => ({ valid: true }) }));
afterEach(() => vi.restoreAllMocks());
const exhausted = () => new Response(JSON.stringify({ ...responseObject('exhausted', 'PARTIAL_CHECKPOINT'),
  status: 'incomplete', incomplete_details: { reason: 'max_output_tokens' },
  usage: { input_tokens: 2000, output_tokens: 8192, output_tokens_details: { reasoning_tokens: 8100 } },
}));
function setup(contextWindow = 64_000) {
  const frames: any[] = [];
  const exec = vi.fn(async (command: string) => ({ stdout: 'observed', stderr: '', exitCode: command === 'inspect_43' ? 1 : 0 }));
  const agent = new AgentCore(new TerminalContext(), frame => frames.push(frame),
    async () => ({ ...aiConfig, context_window: contextWindow }), exec, async () => false);
  return { agent, frames, exec };
}

describe('Long task checkpoint recovery in AgentCore', () => {
  it.each([true, false])('continues after 86 settled steps without command replay (model recovery=%s)', async modelRecovery => {
    const { agent, frames, exec } = setup(256_000);
    const requests: any[] = [];
    let steps = 0;
    let compactions = 0;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      const body = JSON.parse(init!.body as string);
      requests.push(body);
      if (!body.stream) {
        compactions++;
        return compactions === 1 || !modelRecovery ? exhausted()
          : new Response(JSON.stringify(responseObject('checkpoint', JSON.stringify(checkpoint))));
      }
      if (steps++ >= 86) return reply('final', 'Inspection complete.');
      const call = functionCall(`call_${steps}`, 'execute_command', { command: `inspect_${steps}`, timeout_ms: null });
      const response = responseObject(`response_${steps}`, '', [call]);
      response.usage.input_tokens = steps === 86 ? 240_000 : 1000;
      return sse([{ type: 'response.completed', response }]);
    });
    await agent.handleAgentStart('1', 'Inspect the blog and fail2ban; never change firewall rules.');
    expect(compactions).toBe(2);
    expect(frames.at(-1)).toMatchObject({ subType: 'run_end', outcome: 'completed', usage: { checkpointCalls: 2 } });
    expect(frames.filter(frame => frame.subType === 'error')).toEqual([]);
    expect(frames).toContainEqual(expect.objectContaining({ subType: 'context_status', compacted: true,
      checkpointSource: modelRecovery ? 'model' : 'local' }));
    const commands = exec.mock.calls.map(([command]) => command).filter(command => command.startsWith('inspect_'));
    expect(commands).toHaveLength(86);
    expect(new Set(commands).size).toBe(86);
    const next = requests.at(-1);
    expect(next.input.some((item: any) => item.type === 'function_call_output')).toBe(false);
    const factItem = next.input.find((item: any) => item.content?.includes('[UNTRUSTED OBSERVATION: execution facts]'));
    const ledger = JSON.parse(factItem.content.split('\n')[1]);
    expect(ledger.rows).toHaveLength(86);
    expect(ledger.rows.find((row: any[]) => row[2] === 'inspect_43').slice(3)).toEqual(['failed', true, 1]);
    expect(JSON.stringify(next.input)).toContain('never change firewall rules');
    expect(JSON.stringify(requests)).not.toContain('PARTIAL_CHECKPOINT');
  });

  it('rebuilds the unaccepted checkpoint and ledger when the first post-compaction request fails', async () => {
    const { agent, frames, exec } = setup();
    const requests: any[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      const body = JSON.parse(init!.body as string); requests.push(body);
      if (requests.length === 1) {
        const response = responseObject('first', '', [functionCall()]);
        response.usage.input_tokens = 55_000;
        return sse([{ type: 'response.completed', response }]);
      }
      if (!body.stream) return new Response(JSON.stringify(responseObject('cp', JSON.stringify(checkpoint))));
      if (requests.length === 3) return new Response('{"error":{"code":"invalid_api_key"}}', { status: 401 });
      return reply('resumed', 'Current state verified.');
    });
    await agent.handleAgentStart('1', 'Inspect nginx; do not restart it.');
    expect(frames.at(-1)).toMatchObject({ subType: 'run_end', outcome: 'failed' });
    await agent.handleAgentStart('1', 'Continue the interrupted task');
    expect(frames.at(-1)).toMatchObject({ subType: 'run_end', outcome: 'completed' });
    expect(requests).toHaveLength(4);
    expect(JSON.stringify(requests[3].input)).toContain('context checkpoint');
    expect(JSON.stringify(requests[3].input)).toContain('execution facts');
    expect(JSON.stringify(requests[3].input)).toContain('do not restart it');
    expect(JSON.stringify(requests[3].input)).toContain('Continue the interrupted task');
    expect(exec.mock.calls.filter(([command]) => command === 'free -m')).toHaveLength(1);
  });

  it('keeps native history and settled outputs if the complete continuation cannot fit', async () => {
    const { agent, frames } = setup();
    const context = (agent as any).context;
    vi.spyOn(context, 'fitsBudget').mockReturnValue(false);
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      const body = JSON.parse(init!.body as string);
      if (!body.stream) return new Response(JSON.stringify(responseObject('cp', JSON.stringify(checkpoint))));
      const response = responseObject('first', '', [functionCall()]); response.usage.input_tokens = 55_000;
      return sse([{ type: 'response.completed', response }]);
    });
    await agent.handleAgentStart('1', 'Inspect');
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(frames.find(frame => frame.subType === 'error')).toMatchObject({ code: 'responses_budget', diagnostics: { purpose: 'checkpoint' } });
    expect(context.summary).toBe('');
    expect(context.history).toContainEqual(expect.objectContaining({ type: 'function_call', call_id: 'call_1' }));
    expect(context.pending).toContainEqual(expect.objectContaining({ type: 'function_call_output', call_id: 'call_1' }));
  });

  it('reset during checkpoint recovery cannot replace the new session or resume old commands', async () => {
    const { agent, frames, exec } = setup();
    let requests = 0;
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      requests++;
      if (requests === 1) {
        const response = responseObject('first', '', [functionCall()]); response.usage.input_tokens = 55_000;
        return sse([{ type: 'response.completed', response }]);
      }
      if (requests === 2) return exhausted();
      return new Promise((_resolve, reject) => init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true }));
    });
    const pending = agent.handleAgentStart('1', 'Inspect');
    await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(3));
    agent.resetSession();
    await pending;
    expect((agent as any).context.buildInput()).toEqual([]);
    expect(frames.filter(frame => frame.subType === 'context_status' && frame.compacted)).toEqual([]);
    expect(exec.mock.calls.filter(([command]) => command === 'free -m')).toHaveLength(1);
    expect(agent.getStatus()).toBe('idle');
  });
});
