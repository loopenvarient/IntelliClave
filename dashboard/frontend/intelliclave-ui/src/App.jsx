import { useEffect, useState, useRef } from 'react'
import axios from 'axios'
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine
} from 'recharts'
import PredictionsPage from './PredictionsPage'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8001'

// ── Fallback data ─────────────────────────────────────────────────────────────
const DUMMY_ROUNDS = [
  { round: 1,  macro_f1: 0.198, accuracy: 0.230, epsilon: 1.058, loss: 1.132 },
  { round: 2,  macro_f1: 0.238, accuracy: 0.281, epsilon: 1.394, loss: 0.989 },
  { round: 3,  macro_f1: 0.275, accuracy: 0.315, epsilon: 1.659, loss: 0.850 },
  { round: 4,  macro_f1: 0.297, accuracy: 0.344, epsilon: 1.887, loss: 0.766 },
  { round: 5,  macro_f1: 0.327, accuracy: 0.385, epsilon: 2.093, loss: 0.707 },
  { round: 6,  macro_f1: 0.352, accuracy: 0.416, epsilon: 2.283, loss: 0.684 },
  { round: 7,  macro_f1: 0.385, accuracy: 0.468, epsilon: 2.461, loss: 0.678 },
  { round: 8,  macro_f1: 0.409, accuracy: 0.489, epsilon: 2.629, loss: 0.747 },
  { round: 9,  macro_f1: 0.443, accuracy: 0.530, epsilon: 2.789, loss: 0.782 },
  { round: 10, macro_f1: 0.453, accuracy: 0.568, epsilon: 2.942, loss: 0.838 },
  { round: 11, macro_f1: 0.478, accuracy: 0.583, epsilon: 3.090, loss: 0.830 },
  { round: 12, macro_f1: 0.497, accuracy: 0.610, epsilon: 3.233, loss: 0.899 },
  { round: 13, macro_f1: 0.511, accuracy: 0.623, epsilon: 3.371, loss: 0.865 },
  { round: 14, macro_f1: 0.516, accuracy: 0.623, epsilon: 3.505, loss: 0.920 },
  { round: 15, macro_f1: 0.535, accuracy: 0.651, epsilon: 3.636, loss: 0.904 },
  { round: 16, macro_f1: 0.539, accuracy: 0.650, epsilon: 3.764, loss: 0.918 },
  { round: 17, macro_f1: 0.556, accuracy: 0.673, epsilon: 3.888, loss: 0.869 },
  { round: 18, macro_f1: 0.558, accuracy: 0.679, epsilon: 4.010, loss: 0.881 },
  { round: 19, macro_f1: 0.586, accuracy: 0.700, epsilon: 4.130, loss: 0.810 },
  { round: 20, macro_f1: 0.603, accuracy: 0.713, epsilon: 4.247, loss: 0.777 },
]

const DUMMY_STATUS = {
  round: 35, total_rounds: 35, epsilon: 7.995,
  macro_f1: 0.730, accuracy: 0.820, loss: 0.435,
  training_active: false, noise_scale: 0.5, temperature: 1.0,
  clients: [
    { id: 'Client 1', status: 'ready', samples: 4423 },
    { id: 'Client 2', status: 'ready', samples: 1830 },
    { id: 'Client 3', status: 'ready', samples: 1985 },
  ],
}

const DUMMY_ATTESTATION = {
  tee_verified: true, status: 'VERIFIED',
  platform: 'Intel SGX (simulated)', enclave_id: 'intelliclave-enclave-v1',
  integrity_hash: '3f2bbc8c935d5a10', mrenclave: '3f2bbc8c935d5a10dd26c0f6',
  attestation_time: '2026-06-07T10:34:28Z', mode: 'gramine-direct', environment: 'WSL2',
}

const DUMMY_ATTACKS = {
  model_inversion:      { verdict: 'RESISTANT',   avg_cosine_similarity: -0.0083, high_risk_classes: 0, total_classes: 6 },
  membership_inference: { verdict: 'NEAR RANDOM', auc: 0.503 },
  gradient_poisoning:   { verdict: 'LOW IMPACT',  accuracy_drop: 2.1 },
}

const DUMMY_PER_CLASS = [
  { name: 'Class 0', f1: 0.81 }, { name: 'Class 1', f1: 0.78 },
  { name: 'Class 2', f1: 0.72 }, { name: 'Class 3', f1: 0.69 },
  { name: 'Class 4', f1: 0.74 }, { name: 'Class 5', f1: 0.71 },
]

const DUMMY_TEE = [
  { op: 'Inference',  base: 0.41, tee: 0.56 },
  { op: 'Train Step', base: 2.62, tee: 3.54 },
  { op: 'Encrypt',    base: 1.45, tee: 1.96 },
  { op: 'Decrypt',    base: 4.78, tee: 6.45 },
  { op: 'Model Save', base: 2.76, tee: 3.73 },
]

// ── Utility helpers ──────────────────────────────────────────────────────────
function fmt(v, d = 3) {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(d) : '—'
}
function pct(v) {
  const n = Number(v)
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '—'
}
function asPct(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n * 100 : null
}
function normalizeClients(c) {
  if (Array.isArray(c)) return c
  if (typeof c === 'number') return Array.from({ length: c }, (_, i) => ({ id: `Client ${i + 1}`, status: 'ready', samples: 0 }))
  return []
}
function shortVerdict(verdict) {
  if (!verdict) return '—'
  const parts = verdict.split(/\s[—–-]\s/)
  return parts[0]?.trim() || verdict
}
function EmptyChart({ message = 'Waiting for backend data…' }) {
  return (
    <div className="empty-state">
      <span className="empty-icon">◎</span>
      <span>{message}</span>
    </div>
  )
}

// ── Custom Tooltip ────────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label, percentKeys = [] }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">Round {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className="chart-tooltip-row" style={{ color: p.color }}>
          <span>{p.name}</span>
          <span className="chart-tooltip-value">
            {typeof p.value === 'number'
              ? (percentKeys.includes(p.dataKey) ? pct(p.value) : p.value.toFixed(4))
              : p.value}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Animated Counter ──────────────────────────────────────────────────────────
function AnimCounter({ value, decimals = 3, suffix = '' }) {
  const [disp, setDisp] = useState(0)
  const ref = useRef(null)
  useEffect(() => {
    const target = Number(value) || 0
    const startTime = performance.now()
    const tick = (now) => {
      const progress = Math.min((now - startTime) / 1200, 1)
      const ease = 1 - Math.pow(1 - progress, 3)
      setDisp(target * ease)
      if (progress < 1) ref.current = requestAnimationFrame(tick)
    }
    ref.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(ref.current)
  }, [value])
  return <>{disp.toFixed(decimals)}{suffix}</>
}

// ── Pulse dot ─────────────────────────────────────────────────────────────────
function PulseDot({ color = '#00d4ff', active = true }) {
  return (
    <span style={{ display: 'inline-block', position: 'relative', width: 10, height: 10, marginRight: 6 }}>
      <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: color, opacity: active ? 0.4 : 0.2, animation: active ? 'ping 1.5s ease-in-out infinite' : 'none' }} />
      <span style={{ position: 'absolute', inset: 2, borderRadius: '50%', background: color }} />
    </span>
  )
}

// ── Verdict Badge ─────────────────────────────────────────────────────────────
function VerdictBadge({ verdict }) {
  const display = shortVerdict(verdict)
  const v = display.toUpperCase()
  const cls = v.includes('RESIST') || v.includes('RANDOM') || v.includes('LOW')
    ? 'verdict-good'
    : v.includes('MODERATE') ? 'verdict-warn' : 'verdict-bad'
  return <span className={`verdict-badge ${cls}`}>{display}</span>
}

// ── Nav ───────────────────────────────────────────────────────────────────────
const PAGES = [
  { key: 'overview',     label: 'Overview',     icon: '⬡' },
  { key: 'training',     label: 'Training',     icon: '◈' },
  { key: 'privacy',      label: 'Privacy',      icon: '⊕' },
  { key: 'clients',      label: 'Clients',      icon: '⬢' },
  { key: 'evaluation',   label: 'Evaluation',   icon: '⊞' },
  { key: 'tee',          label: 'TEE',          icon: '⊟' },
  { key: 'predictions',  label: 'Predictions',  icon: '⊳' },
]

function Sidebar({ active, onChange, backendUp, authToken, onLogout, onLoginClick }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">IC</div>
        <div>
          <div className="brand-name">IntelliClave</div>
          <div className="brand-sub">FL · DP · SGX</div>
        </div>
      </div>
      <nav className="nav">
        {PAGES.map(p => (
          <button key={p.key} className={`nav-btn${active === p.key ? ' active' : ''}`} onClick={() => onChange(p.key)}>
            <span className="nav-ico">{p.icon}</span>
            <span>{p.label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="conn-status">
          <PulseDot color={backendUp ? '#00ff88' : '#ff3c3c'} active={backendUp} />
          <span style={{ fontSize: 12, color: backendUp ? '#00ff88' : '#ff3c3c' }}>
            {backendUp ? 'Backend live' : 'Offline'}
          </span>
        </div>
        {authToken
          ? <button className="auth-btn" onClick={onLogout}>Sign out</button>
          : <button className="auth-btn accent" onClick={onLoginClick}>Sign in</button>
        }
      </div>
    </aside>
  )
}

// ── KPI Card ──────────────────────────────────────────────────────────────────
function KpiCard({ label, value, sub, accent, decimals = 3, suffix = '' }) {
  const accentColor = { cyan: '#00d4ff', green: '#00ff88', purple: '#a78bfa', amber: '#ffaa00' }[accent] || '#00d4ff'
  return (
    <div className="kpi-card" style={{ '--accent': accentColor }}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={{ color: accentColor }}>
        {value !== null && value !== undefined && Number.isFinite(Number(value))
          ? <AnimCounter value={value} decimals={decimals} suffix={suffix} />
          : '—'}
      </div>
      <div className="kpi-sub">{sub}</div>
      <div className="kpi-glow" style={{ background: accentColor }} />
    </div>
  )
}

// ── Panel wrapper ─────────────────────────────────────────────────────────────
function Panel({ title, tag, accent = '#00d4ff', live = false, children }) {
  return (
    <div className={`panel${live ? ' panel-live' : ''}`}>
      <div className="panel-header">
        <span className="panel-title">{title}</span>
        <div className="panel-tags">
          {live && <span className="panel-tag panel-tag-live">LIVE DATA</span>}
          {tag && <span className="panel-tag" style={{ borderColor: `${accent}40`, color: accent }}>{tag}</span>}
        </div>
      </div>
      {children}
    </div>
  )
}

function PageHeader({ title, desc }) {
  return (
    <div className="page-header">
      <h2 className="page-title">{title}</h2>
      {desc && <p className="page-desc">{desc}</p>}
    </div>
  )
}

// ── Training chart ────────────────────────────────────────────────────────────
function TrainingChart({ rounds, live }) {
  const data = rounds?.length ? rounds : (live ? [] : DUMMY_ROUNDS)
  return (
    <Panel title="Training Performance" tag={`${data.length} rounds`} accent="#00d4ff" live={live && data.length > 0}>
      {!data.length ? <EmptyChart /> : <>
        <div className="chart-legend">
          {[{ label: 'Accuracy', color: '#00d4ff' }, { label: 'Macro-F1', color: '#a78bfa' }, { label: 'Loss', color: '#ff6b6b' }].map(s => (
            <span key={s.label} className="chart-legend-item">
              <span className="chart-legend-line" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="round" tick={{ fill: '#475569', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'rgba(255,255,255,0.08)' }} />
            <YAxis tick={{ fill: '#475569', fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={v => v < 1.5 ? `${(v * 100).toFixed(0)}%` : v.toFixed(2)} />
            <Tooltip content={<CustomTooltip percentKeys={['accuracy', 'macro_f1']} />} />
            <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#00d4ff" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
            <Line type="monotone" dataKey="macro_f1" name="Macro-F1" stroke="#a78bfa" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} strokeDasharray="6 3" />
            <Line type="monotone" dataKey="loss" name="Loss" stroke="#ff6b6b" strokeWidth={2} dot={false} activeDot={{ r: 5 }} strokeDasharray="2 4" />
          </LineChart>
        </ResponsiveContainer>
      </>}
    </Panel>
  )
}

// ── Privacy panel ─────────────────────────────────────────────────────────────
function PrivacyPanel({ status, rounds, live }) {
  const eps = Number(status?.epsilon) || 0
  const target = 10
  const pctUsed = Math.min((eps / target) * 100, 100)
  const complete = !status?.training_active && eps >= target * 0.7
  const barColor = complete ? '#00ff88' : pctUsed < 70 ? '#00d4ff' : pctUsed < 90 ? '#ffaa00' : '#ff3c3c'
  const sourceRounds = rounds?.length ? rounds : (live ? [] : DUMMY_ROUNDS)
  const data = sourceRounds.map(r => ({ round: r.round, epsilon: r.epsilon || 0 }))
  const hasEpsilonCurve = data.some(r => r.epsilon > 0)
  return (
    <Panel title="Differential Privacy Budget" tag="DP-SGD" accent="#a78bfa" live={live && eps > 0}>
      <div style={{ textAlign: 'center', padding: '16px 0 12px' }}>
        <div style={{ fontSize: 42, fontWeight: 800, fontFamily: 'monospace', color: barColor, letterSpacing: '-2px' }}>
          ε = {fmt(eps, 4)}
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          δ = 1/n_train · Target ε ≤ {target}
          {complete && <span style={{ color: '#00ff88', marginLeft: 8 }}>· Budget consumed ✓</span>}
        </div>
      </div>
      <div style={{ margin: '12px 0', background: 'rgba(255,255,255,0.05)', borderRadius: 4, height: 8, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pctUsed}%`, background: barColor, borderRadius: 4, transition: 'width 0.6s ease', boxShadow: `0 0 12px ${barColor}60` }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#475569', marginBottom: 12 }}>
        <span>0</span>
        <span style={{ color: barColor, fontWeight: 600 }}>{pctUsed.toFixed(1)}% consumed</span>
        <span>{target}</span>
      </div>
      {hasEpsilonCurve
        ? <ResponsiveContainer width="100%" height={120}>
            <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
              <defs>
                <linearGradient id="epsGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="round" tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} domain={[0, 12]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={target} stroke="#ffaa00" strokeDasharray="6 3" strokeWidth={1.5} label={{ value: `limit ${target}`, fill: '#ffaa00', fontSize: 10, position: 'insideTopRight' }} />
              <Area type="monotone" dataKey="epsilon" name="ε" stroke="#a78bfa" fill="url(#epsGrad)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        : <EmptyChart message="Privacy curve loads after backend sync" />}
    </Panel>
  )
}

// ── Client panel ──────────────────────────────────────────────────────────────
function ClientPanel({ status, live }) {
  const clients  = normalizeClients(status?.clients || [])
  const dist     = status?.client_distribution
  const distData = dist?.chart_data?.length ? dist.chart_data : (live ? [] : [
    { cls: 'C0', c1: 1494, c2: 27,   c3: 201  },
    { cls: 'C1', c1: 1184, c2: 331,  c3: 29   },
    { cls: 'C2', c1: 401,  c2: 651,  c3: 354  },
    { cls: 'C3', c1: 785,  c2: 238,  c3: 754  },
    { cls: 'C4', c1: 782,  c2: 8,    c3: 1116 },
    { cls: 'C5', c1: 883,  c2: 1033, c3: 28   },
  ])
  const maxKl = Math.max(...clients.map(c => Number(c.kl_divergence) || 0), 0.001)
  return (
    <Panel title="Federated Clients" tag={`${clients.length} nodes`} accent="#00ff88" live={live && clients.some(c => c.samples > 0)}>
      <div className="client-grid">
        {clients.map((c, i) => (
          <div key={c.id || i} className="client-card">
            <div className="client-card-head">
              <span className="client-name">{c.id || `Client ${i + 1}`}</span>
              <PulseDot color="#00ff88" active={c.status === 'ready'} />
            </div>
            <div className="client-samples">{Number(c.samples || 0).toLocaleString()}</div>
            <div className="client-samples-label">samples</div>
            <div className="client-kl">KL div: <span>{fmt(c.kl_divergence, 4)}</span></div>
            <div className="client-kl-bar">
              <div style={{ width: `${((Number(c.kl_divergence) || 0) / maxKl) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="panel-subtitle">Class distribution per client</div>
      {distData.length
        ? <ResponsiveContainer width="100%" height={160}>
            <BarChart data={distData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="cls" tick={{ fill: '#475569', fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Bar dataKey="c1" name="Client 1" fill="#00d4ff" radius={[2,2,0,0]} maxBarSize={20} />
              <Bar dataKey="c2" name="Client 2" fill="#a78bfa" radius={[2,2,0,0]} maxBarSize={20} />
              <Bar dataKey="c3" name="Client 3" fill="#00ff88" radius={[2,2,0,0]} maxBarSize={20} />
            </BarChart>
          </ResponsiveContainer>
        : <EmptyChart message="Client distribution unavailable" />}
    </Panel>
  )
}

// ── Local vs global FL comparison ─────────────────────────────────────────────
function FlComparisonPanel({ comparison, live }) {
  const clients = comparison?.clients || []
  const summary = comparison?.summary
  const chartData = clients.map(c => ({
    name: c.client_name || `Client ${c.client_id}`,
    local_f1: c.local?.macro_f1 ?? 0,
    global_f1: c.global?.macro_f1 ?? 0,
    local_acc: c.local?.accuracy ?? 0,
    global_acc: c.global?.accuracy ?? 0,
    gain_f1: c.gain?.macro_f1_pct ?? 0,
  }))

  const useDp = comparison?.local_training?.use_dp
  const eps = comparison?.local_training?.target_epsilon
  const deltaF1 = summary?.avg_gain?.macro_f1_pct ?? 0
  const deltaColor = deltaF1 >= 0 ? '#00ff88' : '#ffaa00'

  return (
    <Panel
      title="Privacy–Utility: Solo vs Federated"
      tag={summary ? `Δ ${deltaF1 >= 0 ? '+' : ''}${fmt(deltaF1, 1)}pp F1` : 'DP-matched'}
      accent="#a78bfa"
      live={live && clients.length > 0}
    >
      <div className="panel-subtitle">
        Fair comparison: <strong style={{ color: '#94a3b8', fontWeight: 600 }}>local solo</strong> vs{' '}
        <strong style={{ color: '#00d4ff', fontWeight: 600 }}>global FL</strong> on each client&apos;s test split
        {useDp ? ` (both under DP-SGD, ε≈${eps ?? 10})` : ''}.
        {useDp === false && (
          <span style={{ display: 'block', marginTop: 4, color: '#ffaa00' }}>
            Stale data: local baselines were trained without DP. Regenerate with{' '}
            <code style={{ color: '#94a3b8' }}>python fl/compare_local_vs_global.py --dp --epsilon 10 --retrain</code>.
          </span>
        )}
      </div>
      {!clients.length ? (
        <EmptyChart message="Run: python fl/compare_local_vs_global.py --dp --epsilon 10 --retrain" />
      ) : (
        <>
          {summary && (
            <div className="comparison-summary">
              <div className="comparison-stat">
                <span className="comparison-stat-label">Solo local{useDp ? ' (DP)' : ''}</span>
                <span className="comparison-stat-value" style={{ color: '#94a3b8' }}>
                  {pct(summary.local_weighted?.macro_f1)}
                </span>
              </div>
              <div className="comparison-stat">
                <span className="comparison-stat-label">Global FL{useDp ? ' (DP)' : ''}</span>
                <span className="comparison-stat-value" style={{ color: '#00d4ff' }}>
                  {pct(summary.global_weighted?.macro_f1)}
                </span>
              </div>
              <div className="comparison-stat">
                <span className="comparison-stat-label">Δ F1 (FL − solo)</span>
                <span className="comparison-stat-value" style={{ color: deltaColor }}>
                  {deltaF1 >= 0 ? '+' : ''}{fmt(deltaF1, 1)}pp
                </span>
              </div>
              <div className="comparison-stat">
                <span className="comparison-stat-label">FL wins (F1)</span>
                <span className="comparison-stat-value" style={{ color: '#e2e8f0' }}>
                  {summary.clients_where_global_wins_f1}/{summary.total_clients}
                </span>
              </div>
            </div>
          )}
          <div className="chart-legend" style={{ marginTop: 12 }}>
            <span className="chart-legend-item"><span className="chart-legend-line" style={{ background: '#64748b' }} />Solo local{useDp ? ' (DP)' : ''}</span>
            <span className="chart-legend-item"><span className="chart-legend-line" style={{ background: '#00d4ff' }} />Global FL{useDp ? ' (DP)' : ''}</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip percentKeys={['local_f1', 'global_f1', 'local_acc', 'global_acc']} />} />
              <Bar dataKey="local_f1" name="Local F1" fill="#64748b" radius={[3, 3, 0, 0]} maxBarSize={36} />
              <Bar dataKey="global_f1" name="Global F1" fill="#00d4ff" radius={[3, 3, 0, 0]} maxBarSize={36} />
            </BarChart>
          </ResponsiveContainer>
          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Client</th>
                  <th>Solo F1</th>
                  <th>Global F1</th>
                  <th>Δ F1</th>
                </tr>
              </thead>
              <tbody>
                {clients.map(c => (
                  <tr key={c.client_id}>
                    <td>{c.client_name}</td>
                    <td>{pct(c.local?.macro_f1)}</td>
                    <td>{pct(c.global?.macro_f1)}</td>
                    <td style={{ color: (c.gain?.macro_f1_pct ?? 0) >= 0 ? '#00ff88' : '#ff6b6b' }}>
                      {(c.gain?.macro_f1_pct ?? 0) >= 0 ? '+' : ''}{fmt(c.gain?.macro_f1_pct, 1)}pp
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  )
}

// ── Eval panel ────────────────────────────────────────────────────────────────
function EvalPanel({ perClass, attacks, live }) {
  const data = perClass?.length ? perClass : (live ? [] : DUMMY_PER_CLASS)
  const mi   = attacks?.model_inversion
  const mem  = attacks?.membership_inference
  const gp   = attacks?.gradient_poisoning
  const hasAttacks = !!(mi || mem || gp)
  const attackRows = hasAttacks ? [
    { label: 'Model inversion',      verdict: mi?.verdict,  detail: `cos sim: ${fmt(mi?.avg_cosine_similarity, 3)} · high-risk: ${mi?.high_risk_classes ?? 0}/${mi?.total_classes ?? 6} classes` },
    { label: 'Membership inference', verdict: mem?.verdict, detail: `AUC: ${fmt(mem?.auc ?? mem?.avg_auc, 3)} · near-random classifier` },
    { label: 'Gradient poisoning',   verdict: gp?.verdict,  detail: `acc drop: ${fmt(gp?.accuracy_drop_pct ?? (gp?.accuracy_drop != null ? gp.accuracy_drop * 100 : null), 1)}%` },
  ] : (live ? [] : [
    { label: 'Model inversion',      verdict: DUMMY_ATTACKS.model_inversion.verdict,      detail: `cos sim: ${fmt(DUMMY_ATTACKS.model_inversion.avg_cosine_similarity, 4)} · high-risk: 0/6 classes` },
    { label: 'Membership inference', verdict: DUMMY_ATTACKS.membership_inference.verdict, detail: `AUC: ${fmt(DUMMY_ATTACKS.membership_inference.auc, 3)} · near-random classifier` },
    { label: 'Gradient poisoning',   verdict: DUMMY_ATTACKS.gradient_poisoning.verdict,   detail: `acc drop: ${fmt(DUMMY_ATTACKS.gradient_poisoning.accuracy_drop, 1)}%` },
  ])
  return (
    <Panel title="Model Evaluation" tag="6-class" accent="#00ff88" live={live && (data.length > 0 || hasAttacks)}>
      <div className="panel-subtitle">Per-class F1 score (global model)</div>
      {data.length
        ? <ResponsiveContainer width="100%" height={160}>
            <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip percentKeys={['f1']} />} />
              <Bar dataKey="f1" name="F1 Score" fill="#00d4ff" radius={[3,3,0,0]} maxBarSize={40} />
            </BarChart>
          </ResponsiveContainer>
        : <EmptyChart message="Per-class metrics loading…" />}
      <div className="attack-section">
        <div className="panel-subtitle">Security attack evaluation</div>
        {attackRows.length
          ? attackRows.map(a => (
              <div key={a.label} className="attack-row">
                <div>
                  <div className="attack-label">{a.label}</div>
                  <div className="attack-detail">{a.detail}</div>
                </div>
                <VerdictBadge verdict={a.verdict} />
              </div>
            ))
          : <EmptyChart message="Attack results unavailable" />}
      </div>
    </Panel>
  )
}

// ── TEE panel ─────────────────────────────────────────────────────────────────
function TeePanel({ teeData, attestation, live }) {
  const att  = attestation || (live ? {} : DUMMY_ATTESTATION)
  const data = teeData?.length ? teeData : (live ? [] : DUMMY_TEE)
  const ok   = att.tee_verified && att.status === 'VERIFIED'
  return (
    <Panel title="TEE · Intel SGX" tag={att.mode || 'gramine-direct'} accent={ok ? '#00ff88' : '#ff3c3c'} live={live && !!attestation}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 0 16px', borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: 14 }}>
        <div style={{ width: 48, height: 48, borderRadius: 8, background: ok ? 'rgba(0,255,136,0.1)' : 'rgba(255,60,60,0.1)', border: `1px solid ${ok ? 'rgba(0,255,136,0.3)' : 'rgba(255,60,60,0.3)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}>{ok ? '✓' : '✗'}</div>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: ok ? '#00ff88' : '#ff3c3c' }}>{att.status}</div>
          <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>{att.platform}</div>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: '#475569' }}>MRENCLAVE</div>
          <div style={{ fontSize: 11, fontFamily: 'monospace', color: '#94a3b8' }}>{att.integrity_hash}…</div>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
        {[['Enclave ID', att.enclave_id], ['Environment', att.environment],
          ['Attested at', att.attestation_time ? new Date(att.attestation_time).toLocaleString() : '—'], ['Mode', att.mode]
        ].map(([k, v]) => (
          <div key={k} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 6, padding: '8px 10px' }}>
            <div style={{ fontSize: 10, color: '#475569', marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{k}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'monospace', wordBreak: 'break-all' }}>{v || '—'}</div>
          </div>
        ))}
      </div>
      <div className="panel-subtitle">Enclave execution overhead (ms)</div>
      {data.length
        ? <ResponsiveContainer width="100%" height={160}>
            <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="op" tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} unit="ms" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Bar dataKey="base" name="Baseline" fill="#00d4ff" radius={[2,2,0,0]} maxBarSize={24} />
              <Bar dataKey="tee"  name="TEE"      fill="#a78bfa" radius={[2,2,0,0]} maxBarSize={24} />
            </BarChart>
          </ResponsiveContainer>
        : <EmptyChart message="Benchmark data unavailable" />}
    </Panel>
  )
}

// ── System info ───────────────────────────────────────────────────────────────
function SystemPanel({ status }) {
  const nClients = Array.isArray(status?.clients) ? status.clients.length : (status?.clients || 3)
  const rows = [
    ['FL Algorithm',  'FedProx (μ=0.1)'],
    ['Rounds',        `${status?.round ?? '—'} / ${status?.total_rounds ?? '—'}`],
    ['Clients',       `${nClients} federated nodes`],
    ['Noise scale',   `σ = ${status?.noise_scale ?? 0.5}`],
    ['Temperature',   `T = ${status?.temperature ?? 1.0}`],
    ['DP Mechanism',  'DP-SGD (Opacus)'],
    ['Encryption',    'RSA-2048 weight enc.'],
    ['TEE Platform',  'Intel SGX (sim)'],
    ['Model',         `${(status?.model_type || 'mlp').toUpperCase()} (FedProx)`],
    ['Classes',       '6 (non-IID data)'],
    ['Status',        status?.training_active ? '🟢 Training' : '⚪ Complete'],
  ]
  return (
    <Panel title="System Configuration" tag="IntelliClave v1.0" accent="#ffaa00">
      <div style={{ display: 'grid', gap: 1 }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '9px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
            <span style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{k}</span>
            <span style={{ fontSize: 12, color: '#cbd5e1', fontFamily: 'monospace', fontWeight: 600 }}>{v}</span>
          </div>
        ))}
      </div>
    </Panel>
  )
}

// ── Login modal ───────────────────────────────────────────────────────────────
function LoginModal({ onClose, onLogin }) {
  const [user, setUser] = useState('')
  const [pass, setPass] = useState('')
  const [err,  setErr]  = useState('')
  const [loading, setLoading] = useState(false)
  async function submit(e) {
    e.preventDefault(); setLoading(true); setErr('')
    try {
      const params = new URLSearchParams()
      params.append('username', user); params.append('password', pass)
      const res = await axios.post(`${API}/token`, params)
      onLogin(res.data.access_token, res.data.role); onClose()
    } catch { setErr('Invalid credentials') }
    finally   { setLoading(false) }
  }
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' }}>
      <div style={{ background: '#0f1628', border: '1px solid rgba(0,212,255,0.2)', borderRadius: 12, padding: 32, width: 340, position: 'relative', boxShadow: '0 0 60px rgba(0,212,255,0.1)' }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0', marginBottom: 6 }}>Sign in</div>
        <div style={{ fontSize: 13, color: '#64748b', marginBottom: 24 }}>Access the IntelliClave dashboard</div>
        <form onSubmit={submit}>
          <input className="auth-input" placeholder="Username" value={user} onChange={e => setUser(e.target.value)} style={{ marginBottom: 10 }} />
          <input className="auth-input" placeholder="Password" type="password" value={pass} onChange={e => setPass(e.target.value)} style={{ marginBottom: err ? 6 : 16 }} />
          {err && <div style={{ color: '#ff6b6b', fontSize: 12, marginBottom: 12 }}>{err}</div>}
          <button className="auth-submit" type="submit" disabled={loading}>{loading ? 'Signing in…' : 'Sign in'}</button>
        </form>
        <div style={{ marginTop: 16, fontSize: 12, color: '#475569' }}>
          Admin: <code style={{ color: '#00d4ff' }}>admin / adminpass</code><br />
          Viewer: <code style={{ color: '#00d4ff' }}>viewer / viewerpass</code>
        </div>
        <button onClick={onClose} style={{ position: 'absolute', top: 16, right: 16, background: 'none', border: 'none', color: '#475569', cursor: 'pointer', fontSize: 18 }}>×</button>
      </div>
    </div>
  )
}

// ── App CSS extras (appended to App.css at runtime via style tag) ─────────────
const EXTRA_CSS = `
.chart-tooltip{background:rgba(10,14,26,0.97);border:1px solid rgba(0,212,255,0.2);border-radius:8px;padding:10px 14px;font-size:12px;font-family:monospace}
.chart-tooltip-label{color:#64748b;margin-bottom:6px}
.chart-tooltip-row{display:flex;gap:12px;justify-content:space-between}
.chart-tooltip-value{font-weight:700}
.chart-legend{display:flex;gap:20px;margin-bottom:12px}
.chart-legend-item{display:flex;align-items:center;gap:6px;font-size:12px;color:#94a3b8}
.chart-legend-line{width:20px;height:2px;border-radius:2px;display:inline-block}
.panel-subtitle{font-size:12px;color:#64748b;margin-bottom:8px}
.panel-tags{display:flex;gap:6px;align-items:center}
.panel-tag-live{background:rgba(0,255,136,0.08);border-color:rgba(0,255,136,0.3)!important;color:#00ff88!important}
.panel-live{border-color:rgba(0,255,136,0.15)!important}
.empty-state{display:flex;align-items:center;justify-content:center;gap:8px;height:120px;color:#475569;font-size:13px}
.empty-icon{font-size:20px;opacity:0.4}
.client-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
.client-card{background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.12);border-radius:8px;padding:12px 14px}
.client-card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.client-name{font-size:13px;font-weight:600;color:#e2e8f0}
.client-samples{font-size:20px;font-weight:800;color:#00d4ff;font-family:monospace}
.client-samples-label{font-size:11px;color:#475569;margin-bottom:8px}
.client-kl{font-size:11px;color:#64748b;margin-bottom:4px}
.client-kl span{color:#ffaa00}
.client-kl-bar{background:rgba(255,255,255,0.05);border-radius:2px;height:4px}
.client-kl-bar div{height:100%;background:#ffaa00;border-radius:2px;transition:width 0.4s}
.attack-section{margin-top:16px;border-top:1px solid rgba(255,255,255,0.06);padding-top:14px}
.attack-row{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04)}
.attack-label{font-size:13px;color:#cbd5e1}
.attack-detail{font-size:11px;color:#475569;font-family:monospace}
.verdict-badge{padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:.08em;font-family:monospace;border:1px solid}
.verdict-good{background:rgba(0,255,136,0.1);border-color:rgba(0,255,136,0.3);color:#00ff88}
.verdict-warn{background:rgba(255,170,0,0.1);border-color:rgba(255,170,0,0.3);color:#ffaa00}
.verdict-bad{background:rgba(255,60,60,0.1);border-color:rgba(255,60,60,0.3);color:#ff3c3c}
.topbar-left{display:flex;align-items:center;gap:10px}
.topbar-sync{font-size:11px;font-family:monospace;color:#475569;background:rgba(255,255,255,0.03);padding:3px 8px;border-radius:4px}
.topbar-sync.live{color:#00ff88}
.source-badge{font-size:11px;font-family:monospace;color:#64748b;background:rgba(255,255,255,0.02);padding:3px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.06)}
.comparison-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:8px}
.comparison-stat{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:10px 12px}
.comparison-stat-label{display:block;font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.comparison-stat-value{font-size:18px;font-weight:800;font-family:monospace}
.comparison-table-wrap{margin-top:14px;overflow-x:auto}
.comparison-table{width:100%;border-collapse:collapse;font-size:12px}
.comparison-table th{text-align:left;color:#64748b;font-weight:600;padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.08);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.comparison-table td{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.04);color:#cbd5e1;font-family:monospace}
@media(max-width:900px){.comparison-summary{grid-template-columns:repeat(2,1fr)}}
`

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [status,      setStatus]      = useState(null)
  const [rounds,      setRounds]      = useState([])
  const [perClass,    setPerClass]    = useState([])
  const [teeData,     setTeeData]     = useState([])
  const [attestation, setAttestation] = useState(null)
  const [attacks,     setAttacks]     = useState(null)
  const [comparison,  setComparison]  = useState(null)
  const [activePage,  setActivePage]  = useState('overview')
  const [lastPoll,    setLastPoll]    = useState(null)
  const [backendUp,   setBackendUp]   = useState(false)
  const [authToken,   setAuthToken]   = useState(() => localStorage.getItem('ic_token'))
  const [authRole,    setAuthRole]    = useState(() => localStorage.getItem('ic_role'))
  const [showLogin,   setShowLogin]   = useState(false)

  // Inject extra CSS once
  useEffect(() => {
    const tag = document.createElement('style')
    tag.textContent = EXTRA_CSS
    document.head.appendChild(tag)
    return () => document.head.removeChild(tag)
  }, [])

  function handleLogin(token, role) {
    localStorage.setItem('ic_token', token); localStorage.setItem('ic_role', role)
    setAuthToken(token); setAuthRole(role)
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
  }
  function handleLogout() {
    localStorage.removeItem('ic_token'); localStorage.removeItem('ic_role')
    setAuthToken(null); setAuthRole(null)
    delete axios.defaults.headers.common['Authorization']
  }

  useEffect(() => {
    if (authToken) axios.defaults.headers.common['Authorization'] = `Bearer ${authToken}`
    else delete axios.defaults.headers.common['Authorization']

    const poll = async () => {
      const cfg = authToken ? { headers: { Authorization: `Bearer ${authToken}` } } : {}
      try {
        const [s, r, b, a, p, atk, cmp] = await Promise.allSettled([
          axios.get(`${API}/status`, cfg),
          axios.get(`${API}/results`, cfg),
          axios.get(`${API}/benchmarks`, cfg),
          axios.get(`${API}/attestation`, cfg),
          axios.get(`${API}/privacy_log`, cfg),
          axios.get(`${API}/attacks`, cfg),
          axios.get(`${API}/comparison`, cfg),
        ])
        if (s.status === 'fulfilled') {
          const sd = s.value.data
          if (Array.isArray(sd.clients)) {
            sd.clients = sd.clients.map((c, i) => ({ ...c, id: c.id || `Client ${c.client_id || i + 1}` }))
          }
          setStatus(sd); setBackendUp(true); setLastPoll(new Date().toLocaleTimeString())
        } else { setBackendUp(false) }
        if (r.status === 'fulfilled') {
          const rd = r.value.data
          if (rd.rounds?.length) setRounds(rd.rounds)
          if (rd.per_class_f1) {
            setPerClass(Object.entries(rd.per_class_f1)
              .map(([k, v]) => ({ name: k.replace(/_/g, ' '), f1: +Number(v).toFixed(4) }))
              .sort((a, b) => a.name.localeCompare(b.name)))
          }
        }
        if (b.status === 'fulfilled' && b.value.data?.tee_overhead_ms) {
          setTeeData(b.value.data.tee_overhead_ms.map(r => ({
            op:   (r.operation || r.op || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            base: r.baseline_ms,
            tee:  r.tee_ms,
          })))
        }
        if (a.status === 'fulfilled')    setAttestation(a.value.data)
        if (atk.status === 'fulfilled')  setAttacks(atk.value.data)
        if (cmp.status === 'fulfilled')  setComparison(cmp.value.data)
      } catch { setBackendUp(false) }
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [authToken])

  const s    = status || (backendUp ? {} : DUMMY_STATUS)
  const live = backendUp

  const kpiCards = (
    <div className="kpi-row">
      <KpiCard label="Accuracy"  value={asPct(s.accuracy)}  decimals={1} suffix="%" accent="cyan"   sub={`Round ${s.round ?? '—'}/${s.total_rounds ?? '—'}`} />
      <KpiCard label="Macro F1"  value={asPct(s.macro_f1)}  decimals={1} suffix="%" accent="purple" sub="Global model score" />
      <KpiCard label="Loss"      value={s.loss}               decimals={4}           accent="amber"  sub="Cross-entropy loss" />
      <KpiCard label="Privacy ε" value={s.epsilon}            decimals={4}           accent="cyan"   sub={`${(((s.epsilon||0)/10)*100).toFixed(1)}% of ε=10 budget`} />
    </div>
  )

  const pages = {
    overview: <>
      <PageHeader title="System Overview" desc="Full IntelliClave stack — federated learning with differential privacy and TEE attestation." />
      {kpiCards}
      <div className="grid-2"><TrainingChart rounds={rounds} live={live} /><PrivacyPanel status={s} rounds={rounds} live={live} /></div>
      <div className="grid-3"><ClientPanel status={s} live={live} /><EvalPanel perClass={perClass} attacks={attacks} live={live} /><TeePanel teeData={teeData} attestation={attestation} live={live} /></div>
    </>,
    training: <>
      <PageHeader title="FL Training" desc="Round-by-round convergence of the federated global model across 3 non-IID clients." />
      {kpiCards}
      <div className="grid-2"><TrainingChart rounds={rounds} live={live} /><SystemPanel status={s} /></div>
    </>,
    privacy: <>
      <PageHeader title="Privacy Budget" desc="Differential privacy accounting — cumulative ε spend across all FL rounds." />
      {kpiCards}
      <div className="grid-2"><PrivacyPanel status={s} rounds={rounds} live={live} /><EvalPanel perClass={perClass} attacks={attacks} live={live} /></div>
    </>,
    clients: <>
      <PageHeader title="Federated Clients" desc="3-client federation with non-IID class distributions and KL divergence analysis." />
      {kpiCards}
      <ClientPanel status={s} live={live} />
    </>,
    evaluation: <>
      <PageHeader title="Evaluation" desc="Privacy–utility tradeoff (solo vs federated under DP), per-class quality, and security attacks." />
      {kpiCards}
      <FlComparisonPanel comparison={comparison} live={live} />
      <div className="grid-2"><EvalPanel perClass={perClass} attacks={attacks} live={live} /><SystemPanel status={s} /></div>
    </>,
    tee: <>
      <PageHeader title="TEE · Trusted Execution" desc="Intel SGX enclave attestation and execution overhead benchmarks." />
      {kpiCards}
      <TeePanel teeData={teeData} attestation={attestation} live={live} />
    </>,
    predictions: <>
      <PageHeader title="Predictions" desc="Upload a CSV and compare predictions from the global federated model vs individual client local models." />
      {kpiCards}
      <PredictionsPage />
    </>,
  }

  return (
    <div className="app">
      <Sidebar active={activePage} onChange={setActivePage} backendUp={backendUp}
        authToken={authToken} onLogout={handleLogout} onLoginClick={() => setShowLogin(true)} />
      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <span className="topbar-page">{PAGES.find(p => p.key === activePage)?.label}</span>
            {lastPoll && <span className={`topbar-sync${backendUp ? ' live' : ''}`}>{backendUp ? `↻ ${lastPoll}` : '⚠ offline'}</span>}
            {backendUp && s.save_dir && <span className="source-badge" title={s.save_dir}>Run: {s.save_dir.split(/[/\\]/).pop()}</span>}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div className="tee-badge" style={{ color: attestation?.tee_verified ? '#00ff88' : '#ff3c3c', borderColor: attestation?.tee_verified ? 'rgba(0,255,136,0.3)' : 'rgba(255,60,60,0.3)' }}>
              ⬡ {attestation?.tee_verified ? 'TEE VERIFIED' : 'TEE UNVERIFIED'}
            </div>
            {authToken
              ? <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 12, color: '#64748b' }}>{authRole}</span>
                  <button className="auth-btn" onClick={handleLogout}>Sign out</button>
                </div>
              : <button className="auth-btn accent" onClick={() => setShowLogin(true)}>Sign in</button>
            }
          </div>
        </div>
        <div className="content">{pages[activePage] || pages.overview}</div>
      </div>
      {showLogin && <LoginModal onClose={() => setShowLogin(false)} onLogin={handleLogin} />}
    </div>
  )
}