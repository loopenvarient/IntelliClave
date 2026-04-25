import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine
} from 'recharts'
import './App.css'

const API = 'http://localhost:8001'

const DUMMY_ROUNDS = [
  { round: 1, macro_f1: 0.61, accuracy: 0.63, epsilon: 0.22 },
  { round: 2, macro_f1: 0.69, accuracy: 0.71, epsilon: 0.41 },
  { round: 3, macro_f1: 0.74, accuracy: 0.76, epsilon: 0.58 },
  { round: 4, macro_f1: 0.79, accuracy: 0.81, epsilon: 0.77 },
  { round: 5, macro_f1: 0.82, accuracy: 0.83, epsilon: 0.95 },
]

const DUMMY_PER_CLASS = [
  { name: 'WALKING',  f1: 0.91 },
  { name: 'WALK_UP',  f1: 0.84 },
  { name: 'WALK_DN',  f1: 0.79 },
  { name: 'SITTING',  f1: 0.80 },
  { name: 'STANDING', f1: 0.78 },
  { name: 'LAYING',   f1: 0.88 },
]

const DUMMY_TEE = [
  { op: 'encrypt',   base: 12, tee: 18 },
  { op: 'aggregate', base: 45, tee: 61 },
  { op: 'decrypt',   base: 10, tee: 15 },
  { op: 'forward',   base: 8,  tee: 9  },
  { op: 'backward',  base: 22, tee: 27 },
]

const DUMMY_STATUS = {
  round: 5, total_rounds: 20, epsilon: 0.95,
  macro_f1: 0.821, accuracy: 0.834, training_active: false,
  clients: [
    { id: 'Company A', status: 'ready',   samples: 3105 },
    { id: 'Company B', status: 'ready',   samples: 3426 },
    { id: 'Company C', status: 'ready',   samples: 3768 },
  ],
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar() {
  const items = [
    { icon: '◈', label: 'Overview',    active: true  },
    { icon: '⬡', label: 'FL Training', active: false },
    { icon: '⊕', label: 'Privacy',     active: false },
    { icon: '⬢', label: 'Clients',     active: false },
    { icon: '⊞', label: 'Evaluation',  active: false },
    { icon: '⊟', label: 'TEE / Enclave', active: false },
  ]
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>IntelliClave</h1>
        <p>FL + DP Dashboard</p>
      </div>
      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {items.map(i => (
          <div key={i.label} className={`nav-item ${i.active ? 'active' : ''}`}>
            <span className="nav-icon">{i.icon}</span>
            {i.label}
          </div>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span className="dot" />
        Backend connected
      </div>
    </aside>
  )
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────
function KpiRow({ status }) {
  const epsPct = ((status.epsilon / 2.0) * 100).toFixed(1)
  return (
    <div className="kpi-row">
      <div className="kpi-card blue">
        <div className="kpi-label">FL Round</div>
        <div className="kpi-value">{status.round}<span style={{fontSize:'1rem',color:'var(--muted)'}}> / {status.total_rounds}</span></div>
        <div className="kpi-sub">Training progress</div>
      </div>
      <div className="kpi-card green">
        <div className="kpi-label">Macro F1</div>
        <div className="kpi-value">{status.macro_f1.toFixed(3)}</div>
        <div className="kpi-sub">Global model</div>
      </div>
      <div className="kpi-card purple">
        <div className="kpi-label">Accuracy</div>
        <div className="kpi-value">{(status.accuracy * 100).toFixed(1)}%</div>
        <div className="kpi-sub">Test set</div>
      </div>
      <div className="kpi-card amber">
        <div className="kpi-label">Privacy Budget ε</div>
        <div className="kpi-value">{status.epsilon.toFixed(3)}</div>
        <div className="kpi-sub">{epsPct}% of ε=2.0 used</div>
      </div>
    </div>
  )
}

// ── Panel wrapper ─────────────────────────────────────────────────────────────
function Panel({ title, tag, children }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">{title}</span>
        {tag && <span className="panel-tag">{tag}</span>}
      </div>
      {children}
    </div>
  )
}

// ── Training chart ────────────────────────────────────────────────────────────
function TrainingChart({ rounds }) {
  return (
    <Panel title="FL Training Performance" tag="Live">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rounds} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="round" tick={{ fontSize: 11 }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 11 }} />
          <YAxis domain={[0.5, 1]} tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: '#1c2030', border: '1px solid #252a3a', borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="macro_f1" name="Macro F1" stroke="#4f8ef7" strokeWidth={2} dot={{ r: 4 }} />
          <Line type="monotone" dataKey="accuracy"  name="Accuracy"  stroke="#3ecf8e" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// ── Privacy budget ────────────────────────────────────────────────────────────
function PrivacyPanel({ status, rounds }) {
  const eps = status.epsilon
  const target = 2.0
  const pct = Math.min((eps / target) * 100, 100)
  const color = pct < 50 ? 'var(--green)' : pct < 80 ? 'var(--amber)' : 'var(--red)'
  return (
    <Panel title="Privacy Budget" tag="DP-SGD">
      <div className="epsilon-big">
        <div className="value" style={{ color }}>ε = {eps.toFixed(4)}</div>
        <div className="label">δ = 1×10⁻⁵ · Target ε ≤ {target}</div>
      </div>
      <div className="progress-wrap">
        <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="progress-labels">
        <span>0</span>
        <span style={{ color }}>{pct.toFixed(1)}% used</span>
        <span>{target}</span>
      </div>
      <ResponsiveContainer width="100%" height={110} style={{ marginTop: 12 }}>
        <LineChart data={rounds} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="round" tick={{ fontSize: 10 }} />
          <YAxis domain={[0, target]} tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={{ background: '#1c2030', border: '1px solid #252a3a', borderRadius: 8 }} />
          <ReferenceLine y={target} stroke="var(--red)" strokeDasharray="4 4" label={{ value: 'limit', fill: 'var(--red)', fontSize: 10 }} />
          <Line type="monotone" dataKey="epsilon" name="ε" stroke="var(--amber)" strokeWidth={2} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// ── Client status ─────────────────────────────────────────────────────────────
function ClientPanel({ status }) {
  return (
    <Panel title="FL Clients" tag={`${status.clients.length} nodes`}>
      <div className="client-cards">
        {status.clients.map(c => (
          <div key={c.id} className={`client-card ${c.status}`}>
            <div className="client-info">
              <div className="name">{c.id}</div>
              <div className="samples">{c.samples.toLocaleString()} samples</div>
            </div>
            <span className={`client-pill ${c.status}`}>
              {c.status === 'ready' ? '● Ready' : '● Offline'}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  )
}

// ── Per-class F1 ──────────────────────────────────────────────────────────────
function PerClassPanel() {
  return (
    <Panel title="Per-Class F1 Score" tag="HAR · 6 classes">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={DUMMY_PER_CLASS} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 10 }} />
          <YAxis domain={[0.6, 1]} tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={{ background: '#1c2030', border: '1px solid #252a3a', borderRadius: 8 }} />
          <ReferenceLine y={0.8} stroke="var(--muted)" strokeDasharray="4 4" />
          <Bar dataKey="f1" name="F1 Score" fill="#4f8ef7" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// ── TEE overhead ──────────────────────────────────────────────────────────────
function TeePanel() {
  return (
    <Panel title="TEE Overhead" tag="Intel SGX">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={DUMMY_TEE} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="op" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} unit="ms" />
          <Tooltip contentStyle={{ background: '#1c2030', border: '1px solid #252a3a', borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="base" name="Baseline" fill="#3ecf8e" radius={[4, 4, 0, 0]} />
          <Bar dataKey="tee"  name="TEE"      fill="#f5a623" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// ── System info ───────────────────────────────────────────────────────────────
function SystemPanel({ status }) {
  return (
    <Panel title="System Info">
      <table className="stat-table">
        <tbody>
          <tr><td>Dataset</td><td>UCI HAR (10,299 samples)</td></tr>
          <tr><td>Features</td><td>50 (PCA)</td></tr>
          <tr><td>Classes</td><td>6 activities</td></tr>
          <tr><td>FL Algorithm</td><td>FedAvg</td></tr>
          <tr><td>DP Mechanism</td><td>DP-SGD (Opacus)</td></tr>
          <tr><td>TEE Platform</td><td>Intel SGX (sim)</td></tr>
          <tr><td>Status</td><td>{status.training_active ? '🟢 Training' : '⚪ Idle'}</td></tr>
        </tbody>
      </table>
    </Panel>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [status, setStatus] = useState(DUMMY_STATUS)
  const [rounds, setRounds]  = useState(DUMMY_ROUNDS)
  const [lastPoll, setLastPoll] = useState(null)

  useEffect(() => {
    const poll = async () => {
      try {
        const [s, r] = await Promise.all([
          axios.get(`${API}/status`),
          axios.get(`${API}/results`),
        ])
        setStatus(s.data)
        setRounds(r.data.rounds ?? DUMMY_ROUNDS)
        setLastPoll(new Date().toLocaleTimeString())
      } catch { /* backend offline — keep dummy data */ }
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="layout">
      <Sidebar />
      <div className="main">
        <div className="topbar">
          <span className="topbar-title">Overview</span>
          <div className="topbar-right">
            <div className="tee-badge">⬡ TEE VERIFIED</div>
            {lastPoll && <span className="poll-indicator">Last sync {lastPoll}</span>}
          </div>
        </div>
        <div className="content">
          <KpiRow status={status} />
          <div className="grid-2">
            <TrainingChart rounds={rounds} />
            <PrivacyPanel  status={status} rounds={rounds} />
          </div>
          <div className="grid-3">
            <ClientPanel   status={status} />
            <PerClassPanel />
            <TeePanel />
          </div>
          <div className="grid-2" style={{ marginBottom: 0 }}>
            <SystemPanel status={status} />
          </div>
        </div>
      </div>
    </div>
  )
}
