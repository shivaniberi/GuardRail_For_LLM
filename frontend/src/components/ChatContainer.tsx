import { useState, useEffect, useRef } from 'react';
import { ShieldCheck } from 'lucide-react';
import { Sidebar } from './Sidebar';

const API_BASE = 'http://localhost:8000';

// Strip markdown formatting for display in UI panels
function stripMarkdown(text: string): string {
  if (!text || text === '—') return text;
  return text
    .replace(/[*][*](.+?)[*][*]/g, '$1')
    .replace(/[*](.+?)[*]/g, '$1')
    .replace(/^[*]+$/gm, '')
    .replace(/^#{1,6} /gm, '')
    .replace(/^[-+] /gm, '- ')
    .replace(/`(.+?)`/g, '$1')
    .trim();
}

// Strip chain-of-thought artifacts from LLM responses
function cleanResponse(text: string): string {
  if (!text || text === '—') return text;

  // Cut off at chain-of-thought markers
  const cutMarkers = [
  'Chain-of-thought:',
  'Chain-of-Thought:',
  'Chain of thought:',
  'Chain of Thought:',
  'Chain-of-thought',   
  'Chain-of-Thought',   // ← catches capitalized variant
  'Chain of thought',   // ← catches spaced variant
  'Chain of Thought',   // ← catches spaced+capitalized
  '**Reasoning:**',
  'Reasoning:',
  '**Justification:**',
  'Justification:',
  'Assumptions:',
  '**Answer:**',
];
  let cutAt = text.length;
  for (const marker of cutMarkers) {
    const idx = text.toLowerCase().indexOf(marker.toLowerCase());
    if (idx > 0 && idx < cutAt) cutAt = idx;
  }
  let cleaned = text.slice(0, cutAt).trim();

  // Remove inline KB context references sentence by sentence
  const removePatterns = [
    /This can be confirmed from[^.!?]*[.!?]/gi,
    /This is confirmed (by|from|in)[^.!?]*[.!?]/gi,
    /According to the (provided |given )?(context|PRIMARY_KB|WIKIPEDIA_KB|text|information)[^.!?]*[.!?]/gi,
    /The (provided |given )?(context|PRIMARY_KB|WIKIPEDIA_KB|text|information)[^.!?]*(states?|confirms?|says?|mentions?)[^.!?]*[.!?]/gi,
    /This (directly |clearly )?(answers?|confirms?|establishes?)[^.!?]*(question|context|information)[^.!?]*[.!?]/gi,
    /Based on the (provided |given )?(context|PRIMARY_KB|WIKIPEDIA_KB|text|information|historical context)[^.!?]*[.!?]/gi,
    /from the (provided |given )?(context|PRIMARY_KB|WIKIPEDIA_KB|text|information)[^.!?]*[.!?]/gi,
    /as (stated|mentioned|confirmed|established) in[^.!?]*[.!?]/gi,
    /explicitly states?:[^.!?]*[.!?]/gi,
    /\(PRIMARY_KB[^)]*\)/gi,
    /\(WIKIPEDIA_KB[^)]*\)/gi,
  ];

  for (const pattern of removePatterns) {
    cleaned = cleaned.replace(pattern, '');
  }

  // Clean up extra whitespace, double periods, and stray quotes
  cleaned = cleaned.replace(/\s{2,}/g, ' ').replace(/\.\s*\./g, '.').trim();
  cleaned = cleaned.replace(/^["“”]+|["“”]+$/g, '').trim();

  return cleaned;
}

const MODEL_OPTIONS = [
  { key: 'gemma3:1b', label: 'Gemma 3'  },
  { key: 'gemma:2b',  label: 'Gemma'    },
  { key: 'phi3',      label: 'Phi-3'    },
  { key: 'mistral',   label: 'Mistral'  },
  { key: 'qwen2.5',   label: 'Qwen 2.5' },
  { key: 'llama3',    label: 'Llama 3'  },
];

interface Message {
  role: 'user' | 'assistant';
  text: string;
  result?: any;
  mode?: 'single' | 'multi';
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

// ── Single-Agent Guardrail Panel ──────────────────────────────────────────────
function GuardrailAnalysis({ result, theme }: { result: any; theme: 'dark' | 'light' }) {
  const dk = theme === 'dark';
  const cardTitle  = dk ? 'text-gray-400' : 'text-slate-500';
  const cardText   = dk ? 'text-gray-200' : 'text-slate-700';
  const innerBg    = dk ? 'bg-gray-900 text-gray-300' : 'bg-slate-100 text-slate-700';
  const progressBg = dk ? 'bg-gray-700' : 'bg-slate-200';
  const panelBg    = dk ? 'bg-gray-800 border-gray-700' : 'bg-white border-slate-200';
  const card       = dk ? 'bg-gray-900 border border-gray-700' : 'bg-slate-50 border border-slate-200';

  const meta    = result?.metadata || {};
  const out     = result?.guardrails?.output || null;

  // verdict: use rule_based.valid when blocked, else check safety_flags
  const ruleValid  = result?.guardrails?.input?.rule_based?.valid;
  const verdict    = ruleValid !== undefined ? ruleValid : result?.verdict !== 'blocked';

  // Safety checks: backend puts flags in safety_flags{} (key→bool true=triggered)
  // guardrails.input.rule_based.checks is only populated when blocked
  const safetyFlags = result?.safety_flags || {};
  const ruleChecks  = result?.guardrails?.input?.rule_based?.checks || {};
  // Build a unified checks map: passed = flag not triggered
  const checks: Record<string, boolean> = {};
  for (const k of ['privacy', 'hate', 'violence_illegal', 'misinformation', 'bias', 'prompt_injection',
                    'self_harm', 'drug_synthesis', 'financial_fraud', 'ml_unsafe']) {
    if (k in safetyFlags) {
      checks[k] = !safetyFlags[k]; // true=triggered → passed=false
    } else if (k in ruleChecks) {
      checks[k] = ruleChecks[k]?.passed !== false;
    } else {
      checks[k] = true; // not flagged
    }
  }

  // ML probability: backend stores in metadata.ml_unsafe_probability
  const mlProb     = meta?.ml_unsafe_probability ?? result?.guardrails?.input?.ml_based?.unsafe_probability ?? null;
  const mlPrompt   = meta?.ml_prompt_probability ?? null;
  const mlResponse = meta?.ml_response_probability ?? null;

  // Hallucination: backend stores in guardrails.output.checks.hallucination_similarity
  const hallSim = out?.checks?.hallucination_similarity ?? null;

  return (
    <div className={`mt-2 rounded-xl border shadow-lg overflow-hidden ${panelBg}`}>
      <div className={`px-4 py-3 border-b flex items-center gap-2 ${dk ? 'border-gray-700 bg-gray-900' : 'border-slate-200 bg-slate-50'}`}>
        <ShieldCheck size={16} className="text-emerald-500" />
        <span className="text-sm font-semibold">Guardrail Analysis</span>
        <span className={`ml-auto text-xs font-bold px-2 py-0.5 rounded-full ${verdict ? 'bg-emerald-500 text-white' : 'bg-red-500 text-white'}`}>
          {verdict ? 'SAFE' : 'BLOCKED'}
        </span>
      </div>

      <div className="p-4 grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Raw LLM Response</h3>
          <div className={`text-xs leading-relaxed whitespace-pre-wrap overflow-y-auto ${cardText}`} style={{ maxHeight: '220px' }}>
            {stripMarkdown(cleanResponse(
              result?.raw_llm_response ||
              (result?.verdict === 'blocked'
                ? `[Blocked at input — LLM was not called]\nBlock reason: ${result?.block_reason}`
                : '—')
            ))}
          </div>
        </div>
        <div className={`p-3 rounded-lg ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Guarded Response</h3>
          <div className={`text-xs leading-relaxed whitespace-pre-wrap overflow-y-auto ${cardText}`} style={{ maxHeight: '220px' }}>
            {stripMarkdown(cleanResponse(result?.final_response || result?.response || '—'))}
          </div>
        </div>
      </div>

      <div className="px-4 pb-4 grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-3 ${cardTitle}`}>Safety Checks</h3>
          <div className="grid grid-cols-2 gap-2">
            {[
              { key: 'privacy',          label: 'Privacy'          },
              { key: 'hate',             label: 'Hate'             },
              { key: 'violence_illegal', label: 'Violence Illegal' },
              { key: 'misinformation',   label: 'Misinformation'   },
              { key: 'bias',             label: 'Bias'             },
              { key: 'prompt_injection', label: 'Prompt Injection' },
              { key: 'self_harm',        label: 'Self Harm'        },
              { key: 'drug_synthesis',   label: 'Drug Synthesis'   },
              { key: 'financial_fraud',  label: 'Financial Fraud'  },
              { key: 'ml_unsafe',        label: 'ML Unsafe'        },
            ].map(({ key, label }) => {
              const passed = checks[key] !== false;
              return (
                <div key={key} className={`flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium ${innerBg}`}>
                  <span>{label}</span>
                  <span className="text-base">{passed ? '✅' : '❌'}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className={`p-3 rounded-lg ${card} space-y-4`}>
          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>ML Check</h3>
            <p className={`text-xs mb-1.5 ${cardText}`}>
              Unsafe probability (max): <span className="font-semibold">{mlProb != null ? `${(mlProb * 100).toFixed(4)}%` : 'N/A'}</span>
            </p>
            <div className={`w-full h-1.5 rounded-full overflow-hidden ${progressBg}`}>
              <div className="h-1.5 bg-red-500 rounded-full" style={{ width: mlProb != null ? `${Math.min(100, mlProb * 100)}%` : '0%' }} />
            </div>
            <div className={`flex justify-between text-xs mt-1 ${cardTitle}`}>
              <span>0% (Safe)</span><span>20% (Threshold)</span><span>100% (Unsafe)</span>
            </div>
            {(mlPrompt != null || mlResponse != null) && (
              <div className={`mt-2 text-xs space-y-0.5 ${cardText}`}>
                {mlPrompt   != null && <p>Prompt score: <span className="font-semibold">{`${(mlPrompt * 100).toFixed(4)}%`}</span></p>}
                {mlResponse != null && <p>Response score: <span className="font-semibold">{`${(mlResponse * 100).toFixed(4)}%`}</span></p>}
              </div>
            )}
          </div>

          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Output Guardrail</h3>
            <p className={`text-xs mb-1.5 ${cardText}`}>
              Hallucination similarity: <span className="font-semibold">{hallSim != null ? `${(hallSim * 100).toFixed(4)}%` : 'N/A'}</span>
            </p>
            <div className={`w-full h-1.5 rounded-full overflow-hidden ${progressBg}`}>
              <div className="h-1.5 bg-emerald-500 rounded-full" style={{ width: hallSim != null ? `${Math.min(100, hallSim * 100)}%` : '0%' }} />
            </div>
            <div className={`flex justify-between text-xs mt-1 ${cardTitle}`}>
              <span>0% (Hallucination)</span><span>60% (Threshold)</span><span>100% (Grounded)</span>
            </div>
          </div>

          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>RAG Metadata</h3>
            <p className={`text-xs ${cardText}`}>RAG used: <span className="font-semibold">{String(meta?.rag_used ?? false)}</span></p>
            <p className={`text-xs ${cardText}`}>Retrieved docs: <span className="font-semibold">{meta?.retrieved_docs_total ?? 0}</span></p>
            <p className={`text-xs ${cardText}`}>KB sources: <span className="font-semibold">{meta?.kb_sources?.join(', ') || 'none'}</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Multi-Agent Debate Panel ──────────────────────────────────────────────────
function DebateAnalysis({ result, theme }: { result: any; theme: 'dark' | 'light' }) {
  const dk = theme === 'dark';
  const panelBg    = dk ? 'bg-gray-800 border-gray-700' : 'bg-white border-slate-200';
  const card       = dk ? 'bg-gray-900 border-gray-700' : 'bg-slate-50 border-slate-200';
  const cardTitle  = dk ? 'text-gray-400' : 'text-slate-400';
  const cardText   = dk ? 'text-gray-200' : 'text-slate-700';
  const progressBg = dk ? 'bg-gray-700' : 'bg-slate-200';
  const labelColor = dk ? 'text-gray-400' : 'text-slate-500';

  const judge      = result?.judge || {};
  const inputValid = result?.input_validation?.valid;
  const ctxMeta    = result?.context_metadata || {};

  // evaluation{} mirrors run_multi_agent_debate.py — all guardrail values live here
  const ev          = result?.evaluation || {};
  const meta        = ev?.metadata || {};               // evaluation.metadata
  const safetyFlags = ev?.safety_flags || {};           // evaluation.safety_flags
  const outVerify   = ev?.guardrails?.output            // evaluation.guardrails.output
                   || result?.output_verification       // fallback
                   || null;

  // Hallucination: from evaluation.guardrails.output.checks or output_verification
  const hallSim = outVerify?.checks?.hallucination_similarity ?? null;

  // ML probability: from evaluation.metadata
  const mlProb     = meta?.ml_unsafe_probability ?? null;
  const mlPrompt   = meta?.ml_prompt_probability ?? null;
  const mlResponse = meta?.ml_response_probability ?? null;

  // RAW LLM RESPONSE = evaluation.raw_llm_response (unguarded candidate answer)
  const rawLLM = cleanResponse(ev?.raw_llm_response || '—');

  // FINAL GUARDRAIL RESPONSE = evaluation.final_response (after safety checks)
  const guardedResponse = cleanResponse(ev?.final_response || judge?.final_answer || '—');

  // Verdict: from evaluation
  const verdict = ev?.verdict ?? (inputValid !== false ? 'safe' : 'blocked');

  const rounds     = result?.debate_rounds || [];
  const agentCount = rounds[0]?.proposals?.length ?? 0;

  return (
    <div className={`mt-2 rounded-xl border shadow-lg overflow-hidden ${panelBg}`}>

      {/* Header */}
      <div className={`px-4 py-3 border-b flex items-center gap-2 ${dk ? 'border-gray-700 bg-gray-900' : 'border-slate-200 bg-slate-50'}`}>
        <ShieldCheck size={16} className="text-blue-500" />
        <span className="text-sm font-semibold">Guardrail Analysis</span>
        {agentCount > 0 && (
          <span className={`text-xs ${labelColor}`}>— {agentCount} agents · {rounds.length} round{rounds.length !== 1 ? 's' : ''}</span>
        )}
        <span className={`ml-auto text-xs font-bold px-2 py-0.5 rounded-full ${verdict !== 'blocked' ? 'bg-emerald-500 text-white' : 'bg-red-500 text-white'}`}>
          {verdict !== 'blocked' ? 'SAFE' : 'BLOCKED'}
        </span>
      </div>

      {/* Top row: Raw LLM + Guarded Response */}
      <div className="p-4 pb-0 grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg border ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Raw LLM Response</h3>
          <div className={`text-xs leading-relaxed whitespace-pre-wrap overflow-y-auto ${cardText}`} style={{ maxHeight: '220px' }}>
            {stripMarkdown(rawLLM) || '—'}
          </div>
        </div>
        <div className={`p-3 rounded-lg border ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Guarded Response</h3>
          <div className={`text-xs leading-relaxed whitespace-pre-wrap overflow-y-auto ${cardText}`} style={{ maxHeight: '220px' }}>
            {stripMarkdown(guardedResponse) || '—'}
          </div>
        </div>
      </div>

      {/* Bottom row: Safety Checks left, ML/Output/RAG right */}
      <div className="px-4 pb-4 pt-4 grid grid-cols-2 gap-4">

        {/* Left — Safety Checks */}
        <div className={`p-3 rounded-lg border ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-3 ${cardTitle}`}>Safety Checks</h3>
          {(() => {
            const inputChecks = result?.input_validation?.checks || {};
            const items = [
              { key: 'privacy',          label: 'Privacy'          },
              { key: 'hate',             label: 'Hate'             },
              { key: 'violence_illegal', label: 'Violence Illegal' },
              { key: 'misinformation',   label: 'Misinformation'   },
              { key: 'bias',             label: 'Bias'             },
              { key: 'prompt_injection', label: 'Prompt Injection' },
              { key: 'self_harm',        label: 'Self Harm'        },
              { key: 'drug_synthesis',   label: 'Drug Synthesis'   },
              { key: 'financial_fraud',  label: 'Financial Fraud'  },
              { key: 'ml_unsafe',        label: 'ML Unsafe'        },
            ];
            return (
              <div className="grid grid-cols-2 gap-2">
                {items.map(({ key, label }) => {
                  // safetyFlags from evaluation.safety_flags: true=triggered=failed
                  const passed = key in safetyFlags
                    ? !safetyFlags[key]
                    : (inputChecks[key]?.passed !== false);
                  return (
                    <div key={key} className={`flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium ${dk ? 'bg-gray-800' : 'bg-slate-100'} ${cardText}`}>
                      <span>{label}</span>
                      <span className="text-base leading-none">{passed ? '✅' : '❌'}</span>
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </div>

        {/* Right — ML Check + Output Guardrail + RAG Metadata */}
        <div className={`p-3 rounded-lg border space-y-4 ${card}`}>

          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>ML Check</h3>
            <p className={`text-xs mb-1.5 font-semibold ${cardText}`}>
              Unsafe probability (max): {mlProb != null ? `${(mlProb * 100).toFixed(4)}%` : 'N/A'}
            </p>
            <div className={`w-full h-1.5 rounded-full overflow-hidden ${progressBg}`}>
              <div className="h-1.5 bg-red-500 rounded-full" style={{ width: mlProb != null ? `${Math.min(100, mlProb * 100)}%` : '0%' }} />
            </div>
            <div className={`flex justify-between text-xs mt-1 ${labelColor}`}>
              <span>0% (Safe)</span><span>20% (Threshold)</span><span>100% (Unsafe)</span>
            </div>
            {(mlPrompt != null || mlResponse != null) && (
              <div className={`mt-2 text-xs space-y-0.5 ${cardText}`}>
                {mlPrompt   != null && <p>Prompt score: <span className="font-semibold">{`${(mlPrompt * 100).toFixed(4)}%`}</span></p>}
                {mlResponse != null && <p>Response score: <span className="font-semibold">{`${(mlResponse * 100).toFixed(4)}%`}</span></p>}
              </div>
            )}
          </div>

          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Output Guardrail</h3>
            <p className={`text-xs mb-1.5 font-semibold ${cardText}`}>
              Hallucination similarity: {hallSim != null ? `${(hallSim * 100).toFixed(4)}%` : 'N/A'}
            </p>
            <div className={`w-full h-1.5 rounded-full overflow-hidden ${progressBg}`}>
              <div className="h-1.5 bg-emerald-500 rounded-full" style={{ width: hallSim != null ? `${Math.min(100, hallSim * 100)}%` : '0%' }} />
            </div>
            <div className={`flex justify-between text-xs mt-1 ${labelColor}`}>
              <span>0% (Hallucination)</span><span>60% (Threshold)</span><span>100% (Grounded)</span>
            </div>
          </div>

          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>RAG Metadata</h3>
            <p className={`text-xs ${cardText}`}>
              RAG used: <span className="font-semibold">{String(meta?.rag_used ?? result?.context_used ?? false)}</span>
            </p>
            <p className={`text-xs ${cardText}`}>
              Retrieved docs: <span className="font-semibold">{meta?.retrieved_docs_total ?? (ctxMeta.primary_count ?? 0) + (ctxMeta.wiki_count ?? 0)}</span>
            </p>
            <p className={`text-xs ${cardText}`}>
              KB sources: <span className="font-semibold">
                {(meta?.kb_sources ?? [ctxMeta.primary_count > 0 && 'primary', ctxMeta.wiki_count > 0 && 'wiki'].filter(Boolean)).join(', ') || 'none'}
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Chat bubble ───────────────────────────────────────────────────────────────
function ChatBubble({ msg, theme }: { msg: Message; theme: 'dark' | 'light' }) {
  const [showAnalysis, setShowAnalysis] = useState(false);
  const dk = theme === 'dark';

  // Clean at render time so even old messages from localStorage are stripped
  const displayText = msg.role === 'assistant' ? stripMarkdown(cleanResponse(msg.text)) : msg.text;

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end mb-3">
        <div className="relative max-w-xl">
          <div className="bg-emerald-600 text-white text-sm px-4 py-3 rounded-2xl rounded-tr-none shadow leading-relaxed">
            {msg.text}
          </div>
          <div className="absolute top-0 right-0 w-0 h-0"
            style={{ borderLeft: '10px solid #059669', borderBottom: '10px solid transparent', transform: 'translateX(100%)' }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-start mb-3">
      {msg.mode === 'multi' && (
        <span className="mb-1 ml-1 text-xs bg-blue-600 text-white px-2 py-0.5 rounded-full">Multi-Agent Debate</span>
      )}
      <div className="relative max-w-2xl w-full">
        <div className={`text-sm px-4 py-3 rounded-2xl rounded-tl-none shadow leading-relaxed ${dk ? 'bg-gray-700 text-gray-100' : 'bg-white text-slate-800 border border-slate-200'}`}>
          {displayText}
        </div>
        <div className="absolute top-0 left-0 w-0 h-0"
          style={{ borderRight: `10px solid ${dk ? '#374151' : '#ffffff'}`, borderBottom: '10px solid transparent', transform: 'translateX(-100%)' }}
        />
      </div>
      {msg.result && (
        <button onClick={() => setShowAnalysis(v => !v)} className="mt-1 ml-1 text-xs text-emerald-500 hover:text-emerald-400 underline underline-offset-2 transition-colors">
          {showAnalysis ? 'Hide Details' : 'View Details'}
        </button>
      )}
      {showAnalysis && msg.result && (
        <div className="w-full mt-1">
          {msg.mode === 'multi'
            ? <DebateAnalysis result={msg.result} theme={theme} />
            : <GuardrailAnalysis result={msg.result} theme={theme} />
          }
        </div>
      )}
    </div>
  );
}

// ── Main container ────────────────────────────────────────────────────────────
export function ChatContainer() {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [model, setModel] = useState<string>('gemma3:1b');
  const [mode, setMode] = useState<'single' | 'multi'>('single');
  const [numAgents, setNumAgents] = useState(3);
  const [rounds, setRounds] = useState(2);
  const [theme, setTheme] = useState<'dark' | 'light'>(() =>
    (localStorage.getItem('gr_theme') as 'dark' | 'light') || 'dark'
  );
  const bottomRef = useRef<HTMLDivElement>(null);

  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try { const raw = localStorage.getItem('gr_conversations'); return raw ? JSON.parse(raw) : []; }
    catch { return []; }
  });

  const [activeConversationId, setActiveConversationId] = useState<string | null>(() => {
    try { const raw = localStorage.getItem('gr_conversations'); const list = raw ? JSON.parse(raw) : []; return list.length ? list[0].id : null; }
    catch { return null; }
  });

  const dk = theme === 'dark';
  const pageBg   = dk ? 'bg-gray-900 text-gray-100' : 'bg-slate-100 text-slate-900';
  const inputBg  = dk ? 'bg-gray-800 text-gray-100 border-gray-700' : 'bg-white text-slate-900 border-slate-300';
  const selectBg = dk ? 'bg-gray-800 text-gray-100' : 'bg-white text-slate-800 border border-slate-300';
  const moonBtn  = dk ? 'bg-gray-700 hover:bg-gray-600' : 'bg-slate-200 hover:bg-slate-300';
  const chatBg   = dk ? 'bg-gray-900' : 'bg-slate-50';

  useEffect(() => { localStorage.setItem('gr_conversations', JSON.stringify(conversations)); }, [conversations]);
  useEffect(() => { document.documentElement.classList.toggle('dark', dk); localStorage.setItem('gr_theme', theme); }, [theme]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [conversations, activeConversationId]);

  const activeConversation = conversations.find(c => c.id === activeConversationId) ?? null;

  function handleNewConversation() {
    const id = String(Date.now());
    setConversations(p => [{ id, title: 'New Conversation', messages: [] }, ...p]);
    setActiveConversationId(id);
    setPrompt('');
  }
  function handleSelectConversation(id: string) { setActiveConversationId(id); }
  function handleRenameConversation(id: string, title: string) {
    setConversations(p => p.map(c => c.id === id ? { ...c, title } : c));
  }
  function handleDeleteConversation(id: string) {
    setConversations(p => p.filter(c => c.id !== id));
    if (activeConversationId === id) {
      const remaining = conversations.filter(c => c.id !== id);
      setActiveConversationId(remaining.length ? remaining[0].id : null);
    }
  }

  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  };

  const submit = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    setError(null);
    const controller = new AbortController();
    abortControllerRef.current = controller;

    let currentId = activeConversationId;
    if (!currentId) {
      const id = String(Date.now());
      setConversations(p => [{ id, title: prompt.slice(0, 40), messages: [] }, ...p]);
      setActiveConversationId(id);
      currentId = id;
    }

    const sentPrompt = prompt;
    setPrompt('');

    setConversations(prev => prev.map(c =>
      c.id === currentId
        ? { ...c, title: c.title === 'New Conversation' ? sentPrompt.slice(0, 40) : c.title, messages: [...c.messages, { role: 'user', text: sentPrompt }] }
        : c
    ));

    try {
      let data: any;

      if (mode === 'multi') {
        const resp = await fetch(`${API_BASE}/api/multi-agent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: sentPrompt, model, num_agents: numAgents, rounds }),
          signal: controller.signal,
        });
        if (resp.status === 429) { throw new Error('A debate is already running. Please wait for it to finish before sending another.'); }
        if (!resp.ok) { const t = await resp.text().catch(() => resp.statusText); throw new Error(`Server ${resp.status}: ${t}`); }
        data = await resp.json();
        // Response shape mirrors run_multi_agent_debate.py full_record:
        //   evaluation.final_response  — guardrail-processed answer (primary)
        //   evaluation.raw_llm_response — unguarded candidate
        //   candidate.content          — best proposal before evaluation
        //   judge.final_answer         — raw judge output
        const evalFinal    = data?.evaluation?.final_response;
        const evalRaw      = data?.evaluation?.raw_llm_response;
        const candidateRaw = data?.candidate?.content;
        const judgeFinal   = data?.judge?.final_answer;

        const rawAnswer =
          (evalFinal    && typeof evalFinal    === 'string' && evalFinal.trim()    && !evalFinal.startsWith('{'))    ? evalFinal    :
          (evalRaw      && typeof evalRaw      === 'string' && evalRaw.trim()      && !evalRaw.startsWith('{'))      ? evalRaw      :
          (candidateRaw && typeof candidateRaw === 'string' && candidateRaw.trim() && !candidateRaw.startsWith('{')) ? candidateRaw :
          (judgeFinal   && typeof judgeFinal   === 'string' && judgeFinal.trim()   && !judgeFinal.startsWith('{'))   ? judgeFinal   :
          'The agents could not produce a final answer.';

        const finalAnswer = cleanResponse(rawAnswer);
        setConversations(prev => prev.map(c =>
          c.id === currentId
            ? { ...c, messages: [...c.messages, { role: 'assistant', text: finalAnswer, result: data, mode: 'multi' }] }
            : c
        ));
      } else {
        const resp = await fetch(`${API_BASE}/api/guardrail`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: sentPrompt, model }),
          signal: controller.signal,
        });
        if (!resp.ok) { const t = await resp.text().catch(() => resp.statusText); throw new Error(`Server ${resp.status}: ${t}`); }
        data = await resp.json();
        setConversations(prev => prev.map(c =>
          c.id === currentId
            ? { ...c, messages: [...c.messages, { role: 'assistant', text: data.response || data.final_response || '', result: data, mode: 'single' }] }
            : c
        ));
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        setError(e.message || String(e));
      }
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false);
    }
  };

  return (
    <div className={`h-screen flex overflow-hidden ${pageBg}`}>
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
        onRename={handleRenameConversation}
        onDelete={handleDeleteConversation}
        theme={theme}
      />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">

        {/* Header */}
        <header className={`flex-shrink-0 flex items-center justify-between px-6 py-3 border-b ${dk ? 'bg-gray-900 border-gray-700' : 'bg-slate-100 border-slate-200'}`}>
          <div className="flex items-center gap-3">
            <img
              src={dk ? '/logo-dark.png' : '/logo-light.png'}
              alt="GateKeeper Logo"
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                objectFit: 'cover',
                display: 'block',
              }}
            />
            <h1 className="text-xl font-bold">GateKeeper</h1>
          </div>
          <div className="flex items-center gap-3">
            <select value={model} onChange={e => setModel(e.target.value)} className={`p-2 rounded-lg text-sm ${selectBg}`}>
              {MODEL_OPTIONS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
            <button onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} className={`p-2 rounded-lg ${moonBtn}`}>
              {dk
                ? <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
                : <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
              }
            </button>
          </div>
        </header>

        {/* Chat area */}
        <div className={`flex-1 overflow-y-auto px-6 py-4 ${chatBg}`}>
          <div className="max-w-4xl mx-auto">
            {(!activeConversation || activeConversation.messages.length === 0) && (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <img
                  src={dk ? '/logo-dark.png' : '/logo-light.png'}
                  alt="GateKeeper Logo"
                  style={{
                    width: '120px',
                    height: '120px',
                    borderRadius: '50%',
                    objectFit: 'cover',
                    objectPosition: 'center',
                    transform: 'scale(1.4)',
                    marginBottom: '32px',
                  }}
                />
                <h2 className={`text-lg font-semibold mb-1 ${dk ? 'text-white' : 'text-slate-800'}`}>GateKeeper</h2>
                <p className={`text-sm ${dk ? 'text-gray-400' : 'text-slate-500'}`}>Ask anything. Your prompts are protected by multi-layer guardrails.</p>
              </div>
            )}

            {activeConversation?.messages.map((msg, i) => (
              <ChatBubble key={i} msg={msg} theme={theme} />
            ))}

            {isLoading && (
              <div className="flex items-center gap-2 mb-3">
                <div className={`px-4 py-3 rounded-2xl rounded-tl-none text-sm flex items-center gap-2 ${dk ? 'bg-gray-700 text-gray-300' : 'bg-white text-slate-600 border border-slate-200'}`}>
                  <div className="animate-spin w-3 h-3 border-2 border-t-transparent rounded-full border-emerald-500" />
                  {mode === 'multi' ? 'Running debate — this may take a few minutes...' : 'Thinking...'}
                </div>
              </div>
            )}

            {error && <div className="mb-3 p-3 bg-red-600 text-white rounded-lg text-sm">Error: {error}</div>}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input bar */}
        <div className={`flex-shrink-0 border-t px-6 py-4 ${dk ? 'bg-gray-900 border-gray-700' : 'bg-slate-100 border-slate-200'}`}>
          <div className="max-w-4xl mx-auto space-y-2">

            {/* Toggle + multi-agent options */}
            <div className="flex items-center gap-4 flex-wrap">

              {/* Toggle switch */}
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <div
                  onClick={() => setMode(mode === 'multi' ? 'single' : 'multi')}
                  className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${mode === 'multi' ? 'bg-blue-600' : dk ? 'bg-gray-600' : 'bg-slate-300'}`}
                >
                  <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform duration-200 ${mode === 'multi' ? 'translate-x-4' : 'translate-x-0'}`} />
                </div>
                <span className={`text-xs font-medium ${mode === 'multi' ? 'text-blue-400' : dk ? 'text-gray-400' : 'text-slate-500'}`}>
                  Multi-Agent Debate
                </span>
              </label>

              {mode === 'multi' && (
                <>
                  <div className="flex items-center gap-1.5 text-xs">
                    <label className={dk ? 'text-gray-400' : 'text-slate-500'}>Agents:</label>
                    <select value={numAgents} onChange={e => setNumAgents(Number(e.target.value))} className={`px-2 py-1 rounded text-xs ${selectBg}`}>
                      {[2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs">
                    <label className={dk ? 'text-gray-400' : 'text-slate-500'}>Rounds:</label>
                    <select value={rounds} onChange={e => setRounds(Number(e.target.value))} className={`px-2 py-1 rounded text-xs ${selectBg}`}>
                      {[1, 2, 3].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </div>
                  
                </>
              )}
            </div>

            {/* Textarea + Send */}
            <div className="flex gap-3 items-end">
              <textarea
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
                placeholder={mode === 'multi' ? 'Ask a question for multi-agent debate...' : 'Type your prompt... (Enter to send, Shift+Enter for new line)'}
                rows={2}
                className={`flex-1 p-3 rounded-xl border focus:outline-none focus:ring-2 ${mode === 'multi' ? 'focus:ring-blue-500' : 'focus:ring-emerald-500'} text-sm resize-none ${inputBg}`}
              />
              {isLoading ? (
                <button
                  onClick={stopGeneration}
                  className="flex-shrink-0 text-white px-5 py-3 rounded-xl text-sm font-medium bg-red-600 hover:bg-red-700"
                >
                  ⏹ Stop
                </button>
              ) : (
                <button
                  onClick={submit}
                  disabled={!prompt.trim()}
                  className={`flex-shrink-0 disabled:opacity-50 text-white px-5 py-3 rounded-xl text-sm font-medium ${mode === 'multi' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-emerald-600 hover:bg-emerald-700'}`}
                >
                  Send
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}