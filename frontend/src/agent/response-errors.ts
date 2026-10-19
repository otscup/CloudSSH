import { normalizeResponseDiagnostics } from '../../../src/agent-response-diagnostics';
import { t } from '../i18n';

const RESPONSE_ERROR_CODES = [
  'ai_not_configured', 'responses_address', 'responses_storage', 'responses_stateful_required',
  'responses_http', 'responses_chain', 'responses_invalid',
  'responses_incomplete', 'responses_failed', 'responses_stream', 'responses_redirect',
  'responses_schema', 'responses_history', 'responses_config_changed', 'responses_input',
  'responses_budget', 'responses_timeout', 'responses_empty', 'responses_key_required', 'responses_limits',
] as const;

export function responseErrorMessage(code: unknown, status?: number, details?: unknown): string {
  const known = RESPONSE_ERROR_CODES.find(value => value === code) || 'responses_failed';
  const httpStatus = Number.isInteger(status) && status! >= 100 && status! <= 599 ? status! : '';
  const diagnostics = normalizeResponseDiagnostics(details);
  const base = t(`agent.errors.${known}` as Parameters<typeof t>[0], { status: httpStatus });
  const message = diagnostics.purpose === 'checkpoint'
    ? t('agent.checkpointFailure') + '\n' + (known === 'responses_incomplete' ? t('agent.checkpointIncomplete') : base)
    : base;
  if (!Object.keys(diagnostics).length) return message;
  let result = message + '\n' + t('agent.errorDetails', {
    purpose: diagnostics.purpose ? t(`agent.errorPurpose.${diagnostics.purpose}`) : '-',
    stage: diagnostics.stage ? t(`agent.errorStage.${diagnostics.stage}`) : '-',
    parameter: diagnostics.parameter ?? '-', providerCode: diagnostics.providerCode ?? '-',
    requestId: diagnostics.requestId ?? '-',
  });
  if (diagnostics.incompleteReason || diagnostics.requestedOutputTokens != null ||
    diagnostics.outputTokens != null || diagnostics.reasoningTokens != null) {
    result += '\n' + t('agent.errorTokenDetails', {
      reason: diagnostics.incompleteReason ? t(`agent.incompleteReason.${diagnostics.incompleteReason}`) : '-',
      requested: diagnostics.requestedOutputTokens ?? '-', output: diagnostics.outputTokens ?? '-',
      reasoning: diagnostics.reasoningTokens ?? '-',
    });
  }
  return result;
}
