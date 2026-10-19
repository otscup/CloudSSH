import { afterEach, describe, expect, it, vi } from 'vitest';
import { prepareContextCheckpoint, CHECKPOINT_DEADLINE_MS } from '../../../src/worker/agent/checkpoint';
import { AgentContext } from '../../../src/worker/agent/context';
import { bytes } from '../../../src/worker/agent/context-limits';
import type { OperationFact } from '../../../src/worker/agent/execution-journal';
import { ResponsesClient } from '../../../src/worker/agent/responses-client';
import type { ModelResponse } from '../../../src/worker/agent/types';
import { aiConfig, checkpoint, responseObject, functionCall } from './responses-fixtures';

vi.mock('../../../src/worker/agent/ssrf', () => ({ validateBaseUrlWithDNS: async () => ({ valid: true }) }));
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });
const complete = (value: unknown = checkpoint) => new Response(JSON.stringify(responseObject('resp_cp', JSON.stringify(value))));
const incomplete = (reason: unknown = 'max_output_tokens') => new Response(JSON.stringify({
  ...responseObject('resp_inc', 'SECRET_PARTIAL_JSON'), status: 'incomplete', incomplete_details: { reason },
  usage: { input_tokens: 100, output_tokens: 8192, output_tokens_details: { reasoning_tokens: 8000 } },
}));
function setup(cap = 8192, count = 3) {
  const context = new AgentContext();
  context.beginTurn('Inspect nginx; never change firewall rules.');
  const model: ModelResponse = { id: 'accepted', text: 'Inspecting', output: [], calls: [] };
  context.accept(model, context.buildInput());
  const facts: OperationFact[] = Array.from({ length: count }, (_, i) => ({ taskId: 'task', callId: `call_${i}`,
    tool: 'execute_command', description: `inspect_${i}`, status: i === 1 ? 'unknown' : 'succeeded',
    executed: true, exitCode: i === 1 ? -1 : 0, evidence: `observed_${i}`, at: i }));
  const controller = new AbortController();
  const budgets = { context_window: 64_000, max_output_tokens: cap };
  const client = new ResponsesClient({ ...aiConfig, ...budgets });
  const options = { context, client, budgets, facts, signal: controller.signal,
    canContinue: vi.fn((input: Parameters<AgentContext['fitsBudget']>[2]) => context.fitsBudget(64_000 - cap, 'Stable', input)),
    onUsage: vi.fn(), onIncomplete: vi.fn(), onRecovering: vi.fn() };
  return { context, facts, controller, options };
}

describe('Atomic checkpoint preparation and bounded token-limit recovery', () => {
  it('never prepares or falls back while a tool chain still has open calls', async () => {
    const { context, options } = setup();
    const call = functionCall();
    context.accept({ id: 'pending', text: '', calls: [call], output: [call] }, context.buildInput());
    const fetch = vi.spyOn(globalThis, 'fetch');
    await expect(prepareContextCheckpoint(options)).rejects.toMatchObject({ code: 'responses_budget' });
    expect(fetch).not.toHaveBeenCalled();
    expect(options.canContinue).not.toHaveBeenCalled();
  });

  it('uses local facts without an HTTP request when the auxiliary envelope cannot fit', async () => {
    const { options } = setup();
    vi.spyOn(options.client, 'canFit').mockReturnValue(false);
    const fetch = vi.spyOn(globalThis, 'fetch');
    const result = await prepareContextCheckpoint(options);
    expect(result.source).toBe('local');
    expect(JSON.parse(result.safety).rows).toHaveLength(3);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('allocates reasoning-inclusive output above the old 4096 cap and keeps the schema in instructions', async () => {
    const { context, options } = setup();
    const before = context.buildInput();
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(complete());
    const result = await prepareContextCheckpoint(options);
    const body = JSON.parse(fetch.mock.calls[0][1]!.body as string);
    expect(body.max_output_tokens).toBe(8192);
    expect(body.reasoning).toEqual({ effort: 'low' });
    expect(body.instructions).toContain(JSON.stringify(body.text.format.schema));
    expect(body).not.toHaveProperty('tools');
    expect(body.store).toBe(false);
    expect(result.source).toBe('model');
    expect(context.buildInput()).toEqual(before); // preparation itself never changes history
    expect(JSON.parse(result.safety).rows[1]).toEqual(['call_1', 'execute_command', 'inspect_1', 'unknown', true, -1]);
  });

  it('regenerates once with a new key and a smaller evidence view, without echoing partial output', async () => {
    const { context, options } = setup(16_384);
    context.record({ role: 'assistant', content: 'detail'.repeat(1000) });
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(incomplete()).mockResolvedValueOnce(complete());
    expect((await prepareContextCheckpoint(options)).source).toBe('model');
    const bodies = fetch.mock.calls.map(([, init]) => JSON.parse(init!.body as string));
    expect(bodies.map(body => body.max_output_tokens)).toEqual([8192, 16_384]);
    expect(bytes(bodies[1].input)).toBeLessThan(bytes(bodies[0].input));
    expect(JSON.stringify(bodies)).not.toContain('SECRET_PARTIAL_JSON');
    expect(bodies.every(body => body.store === false && !body.previous_response_id && !body.tools)).toBe(true);
    const keys = fetch.mock.calls.map(([, init]) => (init!.headers as Record<string, string>)['Idempotency-Key']);
    expect(new Set(keys).size).toBe(2);
    expect(options.onRecovering).toHaveBeenCalledTimes(1);
    expect(options.onIncomplete).toHaveBeenCalledWith(expect.objectContaining({ incompleteReason: 'max_output_tokens', outputTokens: 8192, reasoningTokens: 8000 }));
  });

  it('uses a local fallback after two explicit output-limit stops, retaining every middle status and prior summary', async () => {
    const { context, options } = setup(8192, 86);
    context.summary = JSON.stringify(checkpoint);
    context.beginTurn('Continue the interrupted task');
    const before = context.buildInput();
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => incomplete());
    const result = await prepareContextCheckpoint(options);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(result).toMatchObject({ source: 'local', summary: JSON.stringify(checkpoint) });
    expect(JSON.parse(result.safety).rows).toHaveLength(86);
    expect(JSON.parse(result.safety).rows[43][2]).toBe('inspect_43');
    const input = context.checkpointSegment(result.summary, result.safety);
    expect(JSON.stringify(input)).toContain('never change firewall rules');
    expect(JSON.stringify(input)).toContain('Continue the interrupted task');
    expect(context.buildInput()).toEqual(before);
  });

  it('never exceeds a user-selected small output cap, including during recovery', async () => {
    const { options } = setup(256);
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => incomplete());
    expect((await prepareContextCheckpoint(options)).source).toBe('local');
    expect(fetch.mock.calls.map(([, init]) => JSON.parse(init!.body as string).max_output_tokens)).toEqual([256, 256]);
  });

  it.each(['content_filter', 'SECRET_UNKNOWN_REASON', null])('does not regenerate or fall back for non-token reasons (%s)', async reason => {
    const { context, options } = setup();
    const before = context.buildInput();
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => incomplete(reason));
    await expect(prepareContextCheckpoint(options)).rejects.toMatchObject({ code: 'responses_incomplete', diagnostics: {
      incompleteReason: reason === 'content_filter' ? 'content_filter' : 'unknown',
    } });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(options.canContinue).not.toHaveBeenCalled();
    expect(context.buildInput()).toEqual(before);
  });

  it.each(['http', 'protocol', 'schema'])('preserves history without recovery on %s failure', async type => {
    const { context, options } = setup();
    const before = context.buildInput();
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => type === 'http'
      ? new Response('{"error":{"message":"SECRET"}}', { status: 401 })
      : type === 'protocol' ? new Response(JSON.stringify({ status: 'incomplete', incomplete_details: { reason: 'max_output_tokens' } })) : complete({}));
    await expect(prepareContextCheckpoint(options)).rejects.toMatchObject({ code: type === 'http' ? 'responses_http' : type === 'protocol' ? 'responses_invalid' : 'responses_schema' });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(options.canContinue).not.toHaveBeenCalled();
    expect(context.buildInput()).toEqual(before);
  });

  it('refuses local fallback atomically if mandatory facts/constraints cannot fit the main request', async () => {
    const { context, options } = setup();
    const before = context.buildInput();
    options.canContinue.mockReturnValue(false);
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => incomplete());
    await expect(prepareContextCheckpoint(options)).rejects.toMatchObject({ code: 'responses_budget', diagnostics: { incompleteReason: 'max_output_tokens' } });
    expect(context.buildInput()).toEqual(before);
  });

  it('does not discard a valid model summary before checking the whole continuation budget', async () => {
    const { context, options } = setup();
    const before = context.buildInput();
    options.canContinue.mockReturnValue(false);
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(complete());
    await expect(prepareContextCheckpoint(options)).rejects.toMatchObject({ code: 'responses_budget' });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(context.buildInput()).toEqual(before);
  });

  it('shares one deadline across both generations and does not recover a stalled response', async () => {
    vi.useFakeTimers();
    const { context, options } = setup();
    const before = context.buildInput();
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementationOnce(async () => {
      await new Promise(resolve => setTimeout(resolve, CHECKPOINT_DEADLINE_MS - 1000));
      return incomplete();
    }).mockImplementation(async (_url, init) => new Promise((_resolve, reject) => {
      init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true });
    }));
    const pending = prepareContextCheckpoint(options);
    const checked = expect(pending).rejects.toMatchObject({ code: 'responses_timeout' });
    await vi.advanceTimersByTimeAsync(CHECKPOINT_DEADLINE_MS - 1000);
    expect(fetch).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1000);
    await checked;
    expect(context.buildInput()).toEqual(before);
  });

  it('cleans the shared deadline after a successful checkpoint', async () => {
    vi.useFakeTimers();
    const { options } = setup();
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(complete());
    await prepareContextCheckpoint(options);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('cancels recovery promptly on stop/reset without local replacement', async () => {
    const { controller, context, options } = setup();
    const before = context.buildInput();
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(incomplete())
      .mockImplementation(async (_url, init) => new Promise((_resolve, reject) => {
        init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true });
      }));
    const pending = prepareContextCheckpoint(options);
    const checked = expect(pending).rejects.toBe('reset');
    await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    controller.abort('reset');
    await checked;
    expect(options.canContinue).not.toHaveBeenCalled();
    expect(context.buildInput()).toEqual(before);
  });
});
