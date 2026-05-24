import { useEffect, useState } from 'react'
import axios from 'axios'
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine
} from 'recharts'
import './App.css'

const API = 'http://localhost:8001'

// ── Fallback data (shown when backend is offline) ─────────────────────────────
const DUMMY_ROUNDS = [
  { round: 1, macro_f1: 0.485,  accuracy: 0.545, epsilon: 4.982 },
  { round: 2, macro_f1: 0.803,  accuracy: 0.808, epsilon: 6.548 },
  { round: 3, macro_f1: 0.841,  accuracy: 0.848, epsilon: 7.826 },
  { round: 4, macro_f1: 0.855,  accuracy: 0.862, epsilon: 8.957 },
  { round: 5, macro_f1: 0.900,  accuracy: 0.901, epsilon: 9.992 },
]

const DUMMY_PER_CLASS = [
  { name: 'WALKING',  f1: 0.994 },
  { name: 'WALK_UP',  f1: 0.988 },
  { name: 'WALK_DN',  f1: 0.989 },
  { name: 'SITTING',  f1: 0.925 },
  { name: 'STANDING', f1: 0.930 },
  { name: 'LAYING',   f1: 1.000 },
]

const DUMMY_TEE = [
  { op: 'inference',  base: 0.41, tee: 0.56 },
  { op: 'train_step', base: 2.62, tee: 3.54 },
  { op: 'encrypt',    base: 1.45, tee: 1.96 },
  { op: 'decrypt',    base: 4.78, tee: 6.45 },
  { op: 'model_save', base: 2.76, tee: 3.73 },
]

const DUMMY_STATUS = {
  round: 5, total_rounds: 20, epsilon: 9.992,
  macro_f1: 0.900, accuracy: 0.901, training_active: false,
  clients: [
    { id: 'FitLife',   status: 'ready', samples: 3105 },
    { id: 'MediTrack', status: 'ready', samples: 3426 },
    { id: 'CareWatch', status: 'ready', samples: 3768 },
  ],
}

const DUMMY_ATTESTATION = {
  tee_verified: true,
  status: 'VERIFIED',
  platform: 'Intel SGX (simulated)',
  enclave_id: 'intelliclave-enclave-v1',
  integrity_hash: '574d8f62a9632715',
  mrenclave: '574d8f62a9632715766d54e1fc27e299a0684220f87f6854b3c135ef345cbcfe',
  attestation_time: '2026-05-03T15:05:08Z',
  mode: 'gramine-direct',
  environment: 'WSL2',
}

const DUMMY_ATTACKS = {
  model_inversion: { verdict: 'RESISTANT', avg_cosine_similarity: 0.31 },
  membership_inference: { verdict: 'NEAR RANDOM', auc: 0.503 },
  gradient_poisoning: { verdict: 'MODERATE', accuracy_drop: 3.78 },
}

const NAV_ITEMS = [
  { key: 'overview', icon: '◈', label: 'Overview', subtitle: 'Dashboard summary' },
  { key: 'training', icon: '⬡', label: 'FL Training', subtitle: 'Round metrics' },
  { key: 'privacy', icon: '⊕', label: 'Privacy', subtitle: 'DP budget' },
  { key: 'clients', icon: '⬢', label: 'Clients', subtitle: 'Data partitions' },
  { key: 'evaluation', icon: '⊞', label: 'Evaluation', subtitle: 'Attack results' },
  { key: 'tee', icon: '⊟', label: 'TEE / Enclave', subtitle: 'Attestation & overhead' },
]

function toFiniteNumber(value) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function formatNumber(value, digits) {
  const numberValue = toFiniteNumber(value)
  return numberValue === null ? '—' : numberValue.toFixed(digits)
}

function normalizeClients(clients) {
  if (Array.isArray(clients)) {
    return clients
  }

  if (typeof clients === 'number' && Number.isFinite(clients)) {
    return Array.from({ length: clients }, (_, index) => ({
      id: `Client ${index + 1}`,
      status: 'ready',
      samples: 0,
    }))
  }

  return []
}

function latestEpsilon(status, rounds, privacyLog) {
  const fromStatus = toFiniteNumber(status?.epsilon)
  if (fromStatus !== null) {
    return fromStatus
  }

  const fromPrivacyLog = Array.isArray(privacyLog) && privacyLog.length > 0
    ? toFiniteNumber(privacyLog[privacyLog.length - 1]?.epsilon)
    : null
  if (fromPrivacyLog !== null) {
    return fromPrivacyLog
  }

  const fromRounds = Array.isArray(rounds) && rounds.length > 0
    ? toFiniteNumber(rounds[rounds.length - 1]?.epsilon)
    : null

  return fromRounds
}

function normalizeStatus(payload, rounds, privacyLog) {
  const normalizedClients = normalizeClients(payload?.clients)
  const roundValue = toFiniteNumber(payload?.round) ?? 0
  const totalRounds = toFiniteNumber(payload?.total_rounds) ?? (Array.isArray(rounds) ? rounds.length : 0)
  const epsilon = latestEpsilon(payload, rounds, privacyLog)
  const macroF1 = toFiniteNumber(payload?.macro_f1)
  const accuracy = toFiniteNumber(payload?.accuracy)

  return {
    ...payload,
    round: roundValue,
    total_rounds: totalRounds,
    clients: normalizedClients,
    macro_f1: macroF1,
    accuracy,
    epsilon,
    training_active: payload?.training_active ?? false,
  }
}

function AttacksPanel({ attacks }) {
  const modelInversion = attacks?.model_inversion ?? DUMMY_ATTACKS.model_inversion
  const membershipInference = attacks?.membership_inference ?? DUMMY_ATTACKS.membership_inference
  const gradientPoisoning = attacks?.gradient_poisoning ?? DUMMY_ATTACKS.gradient_poisoning

  return (
    <Panel title="Security Evaluation" tag="Attacks">
      <table className="stat-table">
        <tbody>
          <tr>
            <td>Model inversion</td>
            <td>{modelInversion.verdict ?? '—'} · cosine {formatNumber(modelInversion.avg_cosine_similarity, 3)}</td>
          </tr>
          <tr>
            <td>Membership inference</td>
            <td>{membershipInference.verdict ?? '—'} · AUC {formatNumber(membershipInference.auc, 3)}</td>
          </tr>
          <tr>
            <td>Gradient poisoning</td>
            <td>{gradientPoisoning.verdict ?? '—'} · drop {formatNumber(gradientPoisoning.accuracy_drop, 2)}%</td>
          </tr>
        </tbody>
      </table>
    </Panel>
  )
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ activePage, onChangePage }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>IntelliClave</h1>
        <p>FL + DP Dashboard</p>
      </div>
      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {NAV_ITEMS.map(item => (
          <button
            key={item.key}
            type="button"
            className={`nav-item ${activePage === item.key ? 'active' : ''}`}
            onClick={() => onChangePage(item.key)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-copy">
              <span className="nav-label">{item.label}</span>
              <span className="nav-subtitle">{item.subtitle}</span>
            </span>
          </button>
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
  const macroF1 = toFiniteNumber(status.macro_f1)
  const accuracy = toFiniteNumber(status.accuracy)
  const epsilon = toFiniteNumber(status.epsilon)
  return (
    <div className="kpi-row">
      <div className="kpi-card blue">
        <div className="kpi-label">FL Round</div>
        <div className="kpi-value">
          {status.round}
          <span style={{ fontSize: '1rem', color: 'var(--muted)' }}> / {status.total_rounds}</span>
        </div>
        <div className="kpi-sub">Training progress</div>
      </div>
      <div className="kpi-card green">
        <div className="kpi-label">Macro F1</div>
        <div className="kpi-value">{formatNumber(macroF1, 3)}</div>
        <div className="kpi-sub">Global model</div>
      </div>
      <div className="kpi-card purple">
        <div className="kpi-label">Accuracy</div>
        <div className="kpi-value">{accuracy === null ? '—' : `${(accuracy * 100).toFixed(1)}%`}</div>
        <div className="kpi-sub">Test set</div>
      </div>
      <div className="kpi-card amber">
        <div className="kpi-label">Privacy Budget ε</div>
        <div className="kpi-value">{formatNumber(epsilon, 3)}</div>
        <div className="kpi-sub">
          {epsilon === null ? 'Privacy budget unavailable' : `${((epsilon / 10.0) * 100).toFixed(1)}% of ε=10 used`}
        </div>
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
          <XAxis
            dataKey="round"
            tick={{ fontSize: 11 }}
            label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 11 }}
          />
          <YAxis domain={[0.4, 1]} tick={{ fontSize: 11 }} />
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
  const eps    = toFiniteNumber(status.epsilon)
  const target = 10.0
  const pct    = eps === null ? 0 : Math.min((eps / target) * 100, 100)

  // Colour logic: green < 70%, amber 70–90%, red > 90%
  // At ε=9.992 (training complete) we show green — budget was correctly consumed
  const isComplete = status.training_active === false && eps !== null && eps >= target * 0.95
  const color = isComplete
    ? 'var(--green)'
    : pct < 70 ? 'var(--green)' : pct < 90 ? 'var(--amber)' : 'var(--red)'

  // Chart Y-axis: add 10% headroom above target so the reference line is visible
  const yMax = Math.ceil(target * 1.12)

  return (
    <Panel title="Privacy Budget" tag="DP-SGD">
      <div className="epsilon-big">
        <div className="value" style={{ color }}>ε = {formatNumber(eps, 4)}</div>
        <div className="label">
          δ = 1/n_train · Target ε ≤ {target}
          {isComplete && <span className="budget-note"> · Training complete ✓</span>}
        </div>
      </div>
      <div className="progress-wrap">
        <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="progress-labels">
        <span>0</span>
        <span style={{ color }}>{pct.toFixed(1)}% consumed</span>
        <span>{target}</span>
      </div>
      <ResponsiveContainer width="100%" height={120} style={{ marginTop: 12 }}>
        <LineChart data={rounds} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="round" tick={{ fontSize: 10 }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 10 }} />
          <YAxis domain={[0, yMax]} tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#1c2030', border: '1px solid #252a3a', borderRadius: 8 }}
            formatter={(v) => [formatNumber(v, 3), 'ε consumed']}
          />
          <ReferenceLine
            y={target}
            stroke="var(--amber)"
            strokeDasharray="6 3"
            strokeWidth={1.5}
            label={{ value: `limit ${target}`, fill: 'var(--amber)', fontSize: 10, position: 'insideTopRight' }}
          />
          <Line
            type="monotone"
            dataKey="epsilon"
            name="ε consumed"
            stroke="#4f8ef7"
            strokeWidth={2}
            dot={{ r: 4, fill: '#4f8ef7' }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// ── Client status ─────────────────────────────────────────────────────────────
function ClientPanel({ status }) {
  const clients = normalizeClients(status.clients)
  return (
    <Panel title="FL Clients" tag={`${clients.length} nodes`}>
      <div className="client-cards">
        {clients.map(c => (
          <div key={c.id} className={`client-card ${c.status}`}>
            <div className="client-info">
              <div className="name">{c.id}</div>
              <div className="samples">{Number.isFinite(Number(c.samples)) ? Number(c.samples).toLocaleString() : '0'} samples</div>
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

function SectionHeader({ title, description }) {
  return (
    <div className="section-header">
      <div>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </div>
  )
}

// ── Per-class F1 ──────────────────────────────────────────────────────────────
function PerClassPanel({ perClass }) {
  const data = perClass && perClass.length > 0 ? perClass : DUMMY_PER_CLASS
  return (
    <Panel title="Per-Class F1 Score" tag="HAR · 6 classes">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" tick={{ fontSize: 10 }} />
          <YAxis domain={[0.85, 1]} tick={{ fontSize: 10 }} tickFormatter={v => v.toFixed(2)} />
          <Tooltip
            contentStyle={{ background: '#1c2030', border: '1px solid #252a3a', borderRadius: 8 }}
            formatter={(v) => [v.toFixed(3), 'F1']}
          />
          <ReferenceLine y={0.9} stroke="var(--muted)" strokeDasharray="4 4" label={{ value: '0.90', fill: 'var(--muted)', fontSize: 9, position: 'insideTopRight' }} />
          <Bar dataKey="f1" name="F1 Score" fill="#4f8ef7" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// ── TEE overhead ──────────────────────────────────────────────────────────────
function TeePanel({ teeData }) {
  const data = teeData && teeData.length > 0 ? teeData : DUMMY_TEE
  return (
    <Panel title="TEE Overhead" tag="Intel SGX">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="op" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} unit="ms" />
          <Tooltip
            contentStyle={{ background: '#1c2030', border: '1px solid #252a3a', borderRadius: 8 }}
            formatter={(v, name) => [`${v.toFixed(2)} ms`, name]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="base" name="Baseline" fill="#3ecf8e" radius={[4, 4, 0, 0]} />
          <Bar dataKey="tee"  name="TEE"      fill="#f5a623" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// ── Attestation panel ─────────────────────────────────────────────────────────
function AttestationPanel({ attestation }) {
  const att      = attestation ?? DUMMY_ATTESTATION
  const verified = att.tee_verified === true && att.status === 'VERIFIED'
  const statusColor = verified ? 'var(--green)' : 'var(--red)'
  const statusIcon  = verified ? '✓' : '✗'

  return (
    <Panel title="TEE Attestation" tag={att.mode ?? 'gramine-direct'}>
      <div className="att-status" style={{ color: statusColor }}>
        <span className="att-icon">{statusIcon}</span>
        <span className="att-label">{att.status}</span>
      </div>
      <table className="stat-table" style={{ marginTop: 12 }}>
        <tbody>
          <tr>
            <td>Platform</td>
            <td>{att.platform}</td>
          </tr>
          <tr>
            <td>Enclave ID</td>
            <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>{att.enclave_id}</td>
          </tr>
          <tr>
            <td>MRENCLAVE</td>
            <td style={{ fontFamily: 'monospace', fontSize: '0.72rem', wordBreak: 'break-all' }}>
              {att.integrity_hash}…
            </td>
          </tr>
          <tr>
            <td>Environment</td>
            <td>{att.environment}</td>
          </tr>
          <tr>
            <td>Attested at</td>
            <td style={{ fontSize: '0.78rem' }}>
              {att.attestation_time
                ? new Date(att.attestation_time).toLocaleString()
                : '—'}
            </td>
          </tr>
        </tbody>
      </table>
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
  const [status,      setStatus]      = useState(DUMMY_STATUS)
  const [rounds,      setRounds]      = useState(DUMMY_ROUNDS)
  const [perClass,    setPerClass]    = useState(DUMMY_PER_CLASS)
  const [teeData,     setTeeData]     = useState(DUMMY_TEE)
  const [attestation, setAttestation] = useState(DUMMY_ATTESTATION)
  const [privacyLog,  setPrivacyLog]  = useState([])
  const [attacks,     setAttacks]     = useState(DUMMY_ATTACKS)
  const [activePage,  setActivePage]  = useState('overview')
  const [lastPoll,    setLastPoll]    = useState(null)
  const [backendUp,   setBackendUp]   = useState(false)
  const [authToken,   setAuthToken]   = useState(localStorage.getItem('intelliclave_token'))
  const [authRole,    setAuthRole]    = useState(localStorage.getItem('intelliclave_role'))
  const [showLogin,   setShowLogin]   = useState(false)
  const [loginUser,   setLoginUser]   = useState('')
  const [loginPass,   setLoginPass]   = useState('')

  useEffect(() => {
    if (authToken) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${authToken}`
    } else {
      delete axios.defaults.headers.common['Authorization']
    }

    const poll = async () => {
      if (!authToken) {
        setBackendUp(false)
        return
      }

      const authConfig = {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      }

      try {
        const [s, r, b, a, p, atk] = await Promise.allSettled([
          axios.get(`${API}/status`, authConfig),
          axios.get(`${API}/results`, authConfig),
          axios.get(`${API}/benchmarks`, authConfig),
          axios.get(`${API}/attestation`, authConfig),
          axios.get(`${API}/privacy_log`, authConfig),
          axios.get(`${API}/attacks`, authConfig),
        ])

        const statusData = s.status === 'fulfilled' ? s.value.data : DUMMY_STATUS
        const roundsData = r.status === 'fulfilled' ? (r.value.data.rounds ?? DUMMY_ROUNDS) : DUMMY_ROUNDS
        const benchmarksData = b.status === 'fulfilled' ? b.value.data : null
        const attestationData = a.status === 'fulfilled' ? a.value.data : DUMMY_ATTESTATION
        const privacyLogData = p.status === 'fulfilled' ? p.value.data : []
        const attacksData = atk.status === 'fulfilled' ? atk.value.data : DUMMY_ATTACKS

        setPrivacyLog(Array.isArray(privacyLogData) ? privacyLogData : [])
        setAttacks(attacksData ?? DUMMY_ATTACKS)
        setStatus(normalizeStatus(statusData, roundsData, privacyLogData))
        setRounds(roundsData)
        setAttestation(attestationData)
        setBackendUp(s.status === 'fulfilled')

        // Per-class F1 — convert object → array for Recharts
        if (r.status === 'fulfilled' && r.value.data.per_class_f1) {
          const SHORT = {
            WALKING: 'WALKING', WALKING_UPSTAIRS: 'WALK_UP',
            WALKING_DOWNSTAIRS: 'WALK_DN', SITTING: 'SITTING',
            STANDING: 'STANDING', LAYING: 'LAYING',
          }
          setPerClass(
            Object.entries(r.value.data.per_class_f1).map(([k, v]) => ({
              name: SHORT[k] ?? k,
              f1: Math.round(v * 1000) / 1000,
            }))
          )
        }

        // TEE overhead — convert array → {op, base, tee} for Recharts
        if (benchmarksData && benchmarksData.tee_overhead_ms) {
          const OP_SHORT = {
            model_inference: 'inference', training_step: 'train',
            encrypt_gradients: 'encrypt', aes_encrypt: 'encrypt',
            aggregate_round: 'aggregate', decrypt_model: 'decrypt',
            aes_decrypt: 'decrypt', forward_pass: 'forward',
            backward_pass: 'backward', model_save: 'save',
          }
          setTeeData(
            benchmarksData.tee_overhead_ms.map(row => ({
              op:   OP_SHORT[row.operation] ?? row.operation,
              base: row.baseline_ms,
              tee:  row.tee_ms,
            }))
          )
        }

        setLastPoll(new Date().toLocaleTimeString())
      } catch {
        setBackendUp(false)
      }
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [authToken])

  async function doLogin(e) {
    e.preventDefault()
    try {
      const params = new URLSearchParams()
      params.append('username', loginUser)
      params.append('password', loginPass)
      const res = await axios.post(`${API}/token`, params)
      const token = res.data.access_token
      const role = res.data.role
      localStorage.setItem('intelliclave_token', token)
      localStorage.setItem('intelliclave_role', role)
      setAuthToken(token)
      setAuthRole(role)
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      setShowLogin(false)
    } catch (err) {
      alert('Login failed')
    }
  }

  function doLogout() {
    localStorage.removeItem('intelliclave_token')
    localStorage.removeItem('intelliclave_role')
    setAuthToken(null)
    setAuthRole(null)
    delete axios.defaults.headers.common['Authorization']
  }

  // TEE badge — wired to live attestation data
  const attVerified = attestation?.tee_verified === true && attestation?.status === 'VERIFIED'
  const teeBadgeColor = attVerified ? 'rgba(62,207,142,0.1)' : 'rgba(224,82,82,0.1)'
  const teeBorderColor = attVerified ? 'rgba(62,207,142,0.3)' : 'rgba(224,82,82,0.3)'
  const teeTextColor = attVerified ? 'var(--green)' : 'var(--red)'
  const teeBadgeText = attVerified ? '⬡ TEE VERIFIED' : '⬡ TEE UNVERIFIED'
  const activeNav = NAV_ITEMS.find(item => item.key === activePage) ?? NAV_ITEMS[0]

  const pageContent = (() => {
    switch (activePage) {
      case 'training':
        return (
          <>
            <SectionHeader title="FL Training" description="Track convergence and model quality across federated rounds." />
            <KpiRow status={status} />
            <div className="grid-2">
              <TrainingChart rounds={rounds} />
              <SystemPanel status={status} />
            </div>
          </>
        )
      case 'privacy':
        return (
          <>
            <SectionHeader title="Privacy" description="Inspect the consumed privacy budget and how it changes over rounds." />
            <KpiRow status={status} />
            <div className="grid-2">
              <PrivacyPanel status={status} rounds={rounds} />
              <AttacksPanel attacks={attacks} />
            </div>
          </>
        )
      case 'clients':
        return (
          <>
            <SectionHeader title="Clients" description="View client participation and class balance for the current run." />
            <KpiRow status={status} />
            <div className="grid-2">
              <ClientPanel status={status} />
              <PerClassPanel perClass={perClass} />
            </div>
          </>
        )
      case 'evaluation':
        return (
          <>
            <SectionHeader title="Evaluation" description="Review class-wise quality and attack resistance." />
            <KpiRow status={status} />
            <div className="grid-2">
              <PerClassPanel perClass={perClass} />
              <AttacksPanel attacks={attacks} />
            </div>
          </>
        )
      case 'tee':
        return (
          <>
            <SectionHeader title="TEE / Enclave" description="Monitor attestation state and the cost of enclave execution." />
            <KpiRow status={status} />
            <div className="grid-2">
              <TeePanel teeData={teeData} />
              <AttestationPanel attestation={attestation} />
            </div>
          </>
        )
      case 'overview':
      default:
        return (
          <>
            <SectionHeader title="Overview" description="A classic control room view for the full IntelliClave stack." />
            <KpiRow status={status} />
            <div className="grid-2">
              <TrainingChart rounds={rounds} />
              <PrivacyPanel status={status} rounds={rounds} />
            </div>
            <div className="grid-3">
              <ClientPanel status={status} />
              <PerClassPanel perClass={perClass} />
              <TeePanel teeData={teeData} />
            </div>
            <div className="grid-2" style={{ marginBottom: 0 }}>
              <AttestationPanel attestation={attestation} />
              <SystemPanel status={status} />
            </div>
          </>
        )
    }
  })()

  return (
    <div className="layout">
      <Sidebar activePage={activePage} onChangePage={setActivePage} />
      <div className="main">
        <div className="topbar">
          <span className="topbar-title">{activeNav.label}</span>
          <div className="topbar-right">
            {/* TEE badge — wired to /attestation API */}
            <div
              className="tee-badge"
              style={{ background: teeBadgeColor, borderColor: teeBorderColor, color: teeTextColor }}
              title={`MRENCLAVE: ${attestation?.integrity_hash ?? '—'}`}
            >
              {teeBadgeText}
            </div>
            {lastPoll && (
              <span className="poll-indicator">
                {backendUp ? `Last sync ${lastPoll}` : 'Backend offline'}
              </span>
            )}
            <div style={{ marginLeft: 12 }}>
              {authToken ? (
                <>
                  <button className="btn" onClick={doLogout}>Logout</button>
                </>
              ) : (
                <>
                  <button className="btn" onClick={() => setShowLogin(!showLogin)}>Login</button>
                </>
              )}
            </div>
            {showLogin && !authToken && (
              <form className="login-form" onSubmit={doLogin} style={{ display: 'flex', gap: 8, alignItems: 'center', marginLeft: 8 }}>
                <input placeholder="username" value={loginUser} onChange={e => setLoginUser(e.target.value)} />
                <input placeholder="password" type="password" value={loginPass} onChange={e => setLoginPass(e.target.value)} />
                <button className="btn" type="submit">OK</button>
              </form>
            )}
          </div>
        </div>
        <div className="content">
          {pageContent}
        </div>
      </div>
    </div>
  )
}
