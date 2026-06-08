/**
 * PredictionsPage.jsx  — fixed version
 *
 * Fixes applied:
 *  1. API reads from VITE_API_URL env var instead of hardcoded localhost
 *  2. Auth token sent on every request (reads from localStorage)
 *  3. Content-Type NOT manually set on FormData (lets browser set boundary)
 *  4. ComparePanel guards against missing accuracy (no label column in CSV)
 *  5. Banner message correct when client outperforms global
 *  6. accDiff / f1Diff sign prefix fixed (no "+-2.3pp")
 *  7. Optional chaining precedence fixed in radar data computation
 */

import { useState, useRef, useCallback } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend,
} from 'recharts'

// Fix 1: read from env var
const API = import.meta.env.VITE_API_URL || 'http://localhost:8001'

// ── Palette ───────────────────────────────────────────────────────────────────
const CLASS_COLORS = ['#00d4ff', '#a78bfa', '#00ff88', '#ffaa00', '#ff6b6b', '#38bdf8']
const MODEL_COLORS = {
  global:   '#00d4ff',
  client_1: '#a78bfa',
  client_2: '#00ff88',
  client_3: '#ffaa00',
}

// Fix 2: auth header helper — reads token App.jsx stores in localStorage
function authHeader() {
  const token = localStorage.getItem('ic_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// ── Shared tooltip ────────────────────────────────────────────────────────────
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'rgba(10,14,26,0.97)', border: '1px solid rgba(0,212,255,0.2)', borderRadius: 8, padding: '10px 14px', fontSize: 12, fontFamily: 'monospace' }}>
      <div style={{ color: '#64748b', marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color || '#e2e8f0', display: 'flex', gap: 12, justifyContent: 'space-between' }}>
          <span>{p.name}</span>
          <span style={{ fontWeight: 700 }}>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
        </div>
      ))}
    </div>
  )
}

// ── Stat box ──────────────────────────────────────────────────────────────────
function StatBox({ label, value, accent = '#00d4ff', sub }) {
  return (
    <div style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${accent}25`, borderTop: `2px solid ${accent}`, borderRadius: 8, padding: '14px 16px', textAlign: 'center' }}>
      <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 800, fontFamily: 'monospace', color: accent }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#475569', marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

// ── Panel wrapper ─────────────────────────────────────────────────────────────
function Panel({ title, tag, accent = '#00d4ff', children }) {
  return (
    <div style={{ background: '#0d1526', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: 20, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9' }}>{title}</span>
        {tag && <span style={{ fontSize: 10, fontWeight: 700, fontFamily: 'monospace', letterSpacing: '0.1em', padding: '3px 8px', border: `1px solid ${accent}40`, borderRadius: 4, color: accent }}>{tag}</span>}
      </div>
      {children}
    </div>
  )
}

// ── Confusion matrix heatmap ──────────────────────────────────────────────────
function ConfusionMatrix({ cm, labels }) {
  if (!cm || !labels) return null
  const matrix = cm.matrix
  const max = Math.max(...matrix.flat())
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>Confusion matrix (rows = true, cols = predicted)</div>
      <table style={{ borderCollapse: 'collapse', fontSize: 11, fontFamily: 'monospace' }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', color: '#475569' }}>T\P</th>
            {labels.map(l => <th key={l} style={{ padding: '4px 8px', color: '#64748b', fontWeight: 600 }}>{l.replace('class_', 'C')}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td style={{ padding: '4px 8px', color: '#64748b', fontWeight: 600 }}>{labels[i]?.replace('class_', 'C')}</td>
              {row.map((val, j) => {
                const intensity = max > 0 ? val / max : 0
                const isCorrect = i === j
                const bg = isCorrect
                  ? `rgba(0,212,255,${0.08 + intensity * 0.5})`
                  : `rgba(255,60,60,${intensity * 0.4})`
                const color = isCorrect ? '#00d4ff' : intensity > 0.3 ? '#ff6b6b' : '#475569'
                return (
                  <td key={j} style={{ padding: '6px 12px', background: bg, border: '1px solid rgba(255,255,255,0.04)', textAlign: 'center', color, fontWeight: isCorrect && val > 0 ? 700 : 400, borderRadius: 2 }}>
                    {val}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Result panel for one model ────────────────────────────────────────────────
function ResultPanel({ result, modelLabel, accent }) {
  const [tab, setTab] = useState('overview')
  if (!result) return null

  const hasLabels = result.accuracy !== undefined
  const tabs = ['overview', 'distribution', 'per-class', ...(hasLabels ? ['confusion'] : []), 'rows']

  return (
    <Panel title={modelLabel} tag={result.model_type?.toUpperCase()} accent={accent}>
      {/* Stat row */}
      <div style={{ display: 'grid', gridTemplateColumns: hasLabels ? 'repeat(4, 1fr)' : 'repeat(2, 1fr)', gap: 10, marginBottom: 16 }}>
        <StatBox label="Rows"     value={result.num_rows}                        accent={accent} />
        {hasLabels && <StatBox label="Accuracy"    value={`${(result.accuracy * 100).toFixed(1)}%`}  accent={accent} sub={`${result.correct}/${result.num_rows} correct`} />}
        {hasLabels && <StatBox label="Macro F1"    value={result.macro_f1?.toFixed(3)}               accent={accent} />}
        {hasLabels && <StatBox label="Weighted F1" value={result.weighted_f1?.toFixed(3)}            accent={accent} />}
        {!hasLabels && <StatBox label="Classes" value={result.num_classes} accent={accent} sub="No true labels in CSV" />}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 14, borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{ padding: '6px 14px', background: tab === t ? `${accent}18` : 'transparent', border: 'none', borderBottom: tab === t ? `2px solid ${accent}` : '2px solid transparent', color: tab === t ? accent : '#64748b', fontFamily: 'monospace', fontSize: 11, fontWeight: 600, cursor: 'pointer', letterSpacing: '0.06em', textTransform: 'uppercase', transition: 'all 0.15s' }}>
            {t}
          </button>
        ))}
      </div>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>Mean confidence per predicted class</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={result.mean_confidence_per_class} margin={{ top: 4, right: 8, bottom: 0, left: -15 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => v.replace('class_', 'C')} />
              <YAxis domain={[0, 1]} tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="mean_confidence" name="Mean confidence" radius={[3, 3, 0, 0]} maxBarSize={50}>
                {result.mean_confidence_per_class.map((_, i) => <Cell key={i} fill={CLASS_COLORS[i % CLASS_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {!hasLabels && (
            <div style={{ marginTop: 12, padding: '10px 14px', background: 'rgba(255,170,0,0.06)', border: '1px solid rgba(255,170,0,0.2)', borderRadius: 6, fontSize: 12, color: '#ffaa00' }}>
              ⚠ CSV has no 'label' column — accuracy and F1 cannot be computed. Add a label column to see full evaluation metrics.
            </div>
          )}
        </div>
      )}

      {/* Distribution tab */}
      {tab === 'distribution' && (
        <div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>Prediction count per class</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={result.prediction_distribution} margin={{ top: 4, right: 8, bottom: 0, left: -15 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => v.replace('class_', 'C')} />
              <YAxis tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="count" name="Predicted count" radius={[3, 3, 0, 0]} maxBarSize={50}>
                {result.prediction_distribution.map((_, i) => <Cell key={i} fill={CLASS_COLORS[i % CLASS_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginTop: 12 }}>
            {result.prediction_distribution.map((d, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: 'rgba(255,255,255,0.02)', borderRadius: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: CLASS_COLORS[i % CLASS_COLORS.length], flexShrink: 0 }} />
                <span style={{ fontSize: 11, color: '#94a3b8' }}>{d.label.replace('class_', 'Class ')}</span>
                <span style={{ marginLeft: 'auto', fontSize: 11, fontFamily: 'monospace', color: CLASS_COLORS[i % CLASS_COLORS.length], fontWeight: 700 }}>{d.pct}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-class F1 tab */}
      {tab === 'per-class' && hasLabels && (
        <div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>F1 score per class</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={result.per_class_f1} margin={{ top: 4, right: 8, bottom: 0, left: -15 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => v.replace('class_', 'C')} />
              <YAxis domain={[0, 1]} tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="f1" name="F1 Score" radius={[3, 3, 0, 0]} maxBarSize={50}>
                {result.per_class_f1.map((d, i) => <Cell key={i} fill={d.f1 > 0.7 ? '#00ff88' : d.f1 > 0.4 ? '#ffaa00' : '#ff6b6b'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      {tab === 'per-class' && !hasLabels && (
        <div style={{ color: '#64748b', fontSize: 13 }}>Add a 'label' column to your CSV to see per-class F1 scores.</div>
      )}

      {/* Confusion matrix tab */}
      {tab === 'confusion' && hasLabels && result.confusion_matrix && (
        <ConfusionMatrix cm={result.confusion_matrix} labels={result.confusion_matrix.labels} />
      )}

      {/* Rows tab */}
      {tab === 'rows' && (
        <div style={{ overflowX: 'auto', maxHeight: 320, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: 'monospace' }}>
            <thead style={{ position: 'sticky', top: 0, background: '#0d1526' }}>
              <tr>
                {['Row', 'Predicted', 'Confidence', ...(hasLabels ? ['True', 'Correct'] : [])].map(h => (
                  <th key={h} style={{ padding: '8px 12px', color: '#475569', fontWeight: 600, textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                  <td style={{ padding: '6px 12px', color: '#475569' }}>{r.row}</td>
                  <td style={{ padding: '6px 12px', color: CLASS_COLORS[r.predicted_class % CLASS_COLORS.length] }}>{r.predicted_label}</td>
                  <td style={{ padding: '6px 12px', color: r.confidence > 0.7 ? '#00ff88' : r.confidence > 0.4 ? '#ffaa00' : '#ff6b6b' }}>{(r.confidence * 100).toFixed(1)}%</td>
                  {hasLabels && <td style={{ padding: '6px 12px', color: '#94a3b8' }}>{r.true_label}</td>}
                  {hasLabels && <td style={{ padding: '6px 12px' }}>{r.correct ? <span style={{ color: '#00ff88' }}>✓</span> : <span style={{ color: '#ff6b6b' }}>✗</span>}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

// ── Comparison panel ──────────────────────────────────────────────────────────
function ComparePanel({ globalResult, clientResult, clientId }) {
  if (!globalResult || !clientResult) return null
  if (!globalResult.per_class_f1 || !clientResult.per_class_f1) return null

  // Fix 4: guard — both CSVs must have had a label column
  if (globalResult.accuracy == null || clientResult.accuracy == null) {
    return (
      <Panel title={`Comparison: Global vs Client ${clientId}`} tag="FEDERATION PROOF" accent="#00d4ff">
        <div style={{ padding: '10px 14px', background: 'rgba(255,170,0,0.06)', border: '1px solid rgba(255,170,0,0.2)', borderRadius: 6, fontSize: 13, color: '#ffaa00' }}>
          ⚠ Upload CSVs with a 'label' column on both sides to enable the full comparison view.
        </div>
      </Panel>
    )
  }

  // Fix 7: correct operator precedence — use ?? instead of ||, wrap before *
  const radarData = globalResult.per_class_f1.map((g, i) => ({
    cls:    g.label.replace('class_', 'C'),
    Global: +((g.f1 ?? 0) * 100).toFixed(1),
    [`Client ${clientId}`]: +((clientResult.per_class_f1[i]?.f1 ?? 0) * 100).toFixed(1),
  }))

  const barData = [
    { metric: 'Accuracy',    Global: +(globalResult.accuracy    * 100).toFixed(1), [`Client ${clientId}`]: +(clientResult.accuracy    * 100).toFixed(1) },
    { metric: 'Macro F1',    Global: +(globalResult.macro_f1    * 100).toFixed(1), [`Client ${clientId}`]: +(clientResult.macro_f1    * 100).toFixed(1) },
    { metric: 'Weighted F1', Global: +(globalResult.weighted_f1 * 100).toFixed(1), [`Client ${clientId}`]: +(clientResult.weighted_f1 * 100).toFixed(1) },
  ]

  const globalWins = globalResult.per_class_f1.filter((g, i) => g.f1 > (clientResult.per_class_f1[i]?.f1 ?? 0)).length
  const accDiff    = ((globalResult.accuracy - clientResult.accuracy) * 100).toFixed(1)
  const f1Diff     = ((globalResult.macro_f1  - clientResult.macro_f1)  * 100).toFixed(1)

  // Fix 5: correct banner message for all three cases
  const bannerText = Number(accDiff) > 0.5
    ? `✓ Global model outperforms Client ${clientId}'s local model`
    : Number(accDiff) < -0.5
      ? `⚠ Client ${clientId}'s local model outperforms global on this CSV`
      : `≈ Global and Client ${clientId} perform similarly on this CSV`

  const bannerColor = Number(accDiff) > 0.5 ? '#00ff88' : Number(accDiff) < -0.5 ? '#ff6b6b' : '#ffaa00'
  const bannerBg    = Number(accDiff) > 0.5 ? 'rgba(0,255,136,0.06)' : Number(accDiff) < -0.5 ? 'rgba(255,60,60,0.06)' : 'rgba(255,170,0,0.06)'
  const bannerBorder= Number(accDiff) > 0.5 ? 'rgba(0,255,136,0.2)'  : Number(accDiff) < -0.5 ? 'rgba(255,60,60,0.2)'  : 'rgba(255,170,0,0.2)'

  // Fix 6: conditional sign prefix — never show "+-N"
  const signedAccDiff = `${Number(accDiff) > 0 ? '+' : ''}${accDiff}`
  const signedF1Diff  = `${Number(f1Diff)  > 0 ? '+' : ''}${f1Diff}`

  return (
    <Panel title={`Comparison: Global vs Client ${clientId}`} tag="FEDERATION PROOF" accent="#00d4ff">
      {/* Key insight banner */}
      <div style={{ padding: '12px 16px', background: bannerBg, border: `1px solid ${bannerBorder}`, borderRadius: 8, marginBottom: 16 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: bannerColor, marginBottom: 4 }}>
          {bannerText}
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>
          Accuracy: <span style={{ color: '#00d4ff', fontFamily: 'monospace' }}>{signedAccDiff}pp</span> ·
          Macro F1: <span style={{ color: '#a78bfa', fontFamily: 'monospace' }}>{signedF1Diff}pp</span> ·
          Global wins <span style={{ color: '#00ff88', fontFamily: 'monospace' }}>{globalWins}/{globalResult.per_class_f1.length}</span> per-class F1 matchups
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Bar chart */}
        <div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>Overall metrics (%)</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="metric" tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#475569', fontSize: 10 }} tickLine={false} axisLine={false} unit="%" />
              <Tooltip content={<ChartTip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Bar dataKey="Global" fill="#00d4ff" radius={[2, 2, 0, 0]} maxBarSize={30} />
              <Bar dataKey={`Client ${clientId}`} fill={MODEL_COLORS[`client_${clientId}`] || '#a78bfa'} radius={[2, 2, 0, 0]} maxBarSize={30} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar chart */}
        <div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>Per-class F1 radar</div>
          <ResponsiveContainer width="100%" height={180}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.08)" />
              <PolarAngleAxis dataKey="cls" tick={{ fill: '#475569', fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#475569', fontSize: 9 }} />
              <Radar name="Global" dataKey="Global" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.15} strokeWidth={2} />
              <Radar name={`Client ${clientId}`} dataKey={`Client ${clientId}`} stroke={MODEL_COLORS[`client_${clientId}`] || '#a78bfa'} fill={MODEL_COLORS[`client_${clientId}`] || '#a78bfa'} fillOpacity={0.1} strokeWidth={2} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-class F1 table */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>Per-class F1 breakdown — where federation helps</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'monospace' }}>
          <thead>
            <tr>
              {['Class', 'Global', `Client ${clientId}`, 'Δ Diff', 'Winner'].map(h => (
                <th key={h} style={{ padding: '8px 12px', color: '#475569', fontWeight: 600, textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {globalResult.per_class_f1.map((g, i) => {
              const c    = clientResult.per_class_f1[i]?.f1 ?? 0
              const diff = g.f1 - c
              const gWin = diff > 0
              return (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                  <td style={{ padding: '8px 12px', color: CLASS_COLORS[i % CLASS_COLORS.length] }}>{g.label.replace('class_', 'Class ')}</td>
                  <td style={{ padding: '8px 12px', color: '#00d4ff',  fontWeight: 600 }}>{(g.f1 * 100).toFixed(1)}%</td>
                  <td style={{ padding: '8px 12px', color: MODEL_COLORS[`client_${clientId}`] || '#a78bfa' }}>{(c * 100).toFixed(1)}%</td>
                  <td style={{ padding: '8px 12px', color: diff > 0 ? '#00ff88' : diff < 0 ? '#ff6b6b' : '#64748b', fontWeight: 600 }}>
                    {diff > 0 ? '+' : ''}{(diff * 100).toFixed(1)}pp
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700, background: gWin ? 'rgba(0,212,255,0.1)' : 'rgba(167,139,250,0.1)', color: gWin ? '#00d4ff' : MODEL_COLORS[`client_${clientId}`] || '#a78bfa' }}>
                      {gWin ? '⬡ GLOBAL' : `CLIENT ${clientId}`}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}

// ── Drop zone ─────────────────────────────────────────────────────────────────
function DropZone({ onFile, loading, label, accent = '#00d4ff' }) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback(e => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file?.name.endsWith('.csv')) onFile(file)
  }, [onFile])

  return (
    <div
      onDrop={handleDrop}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onClick={() => !loading && inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragging ? accent : 'rgba(255,255,255,0.12)'}`,
        borderRadius: 10, padding: '28px 20px', textAlign: 'center',
        cursor: loading ? 'wait' : 'pointer',
        background: dragging ? `${accent}08` : 'rgba(255,255,255,0.02)',
        transition: 'all 0.2s',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        style={{ display: 'none' }}
        onChange={e => e.target.files[0] && onFile(e.target.files[0])}
      />
      <div style={{ fontSize: 28, marginBottom: 8 }}>{loading ? '⟳' : '⬆'}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', marginBottom: 4 }}>
        {loading ? 'Running predictions…' : label}
      </div>
      <div style={{ fontSize: 11, color: '#475569' }}>
        {loading ? 'Please wait' : 'Drop CSV here or click to browse · feature columns + optional label column'}
      </div>
    </div>
  )
}

// ── Main predictions page ─────────────────────────────────────────────────────
export default function PredictionsPage() {
  const [globalResult,   setGlobalResult]   = useState(null)
  const [clientResult,   setClientResult]   = useState(null)
  const [selectedClient, setSelectedClient] = useState(1)
  const [globalLoading,  setGlobalLoading]  = useState(false)
  const [clientLoading,  setClientLoading]  = useState(false)
  const [globalError,    setGlobalError]    = useState('')
  const [clientError,    setClientError]    = useState('')
  const [globalFile,     setGlobalFile]     = useState(null)
  const [clientFile,     setClientFile]     = useState(null)

  // Fix 2 + Fix 3: auth header included, Content-Type NOT manually set
  async function runGlobal(file) {
    setGlobalFile(file); setGlobalLoading(true); setGlobalError(''); setGlobalResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post(`${API}/predict_csv`, form, {
        headers: { ...authHeader() },   // browser sets multipart boundary automatically
      })
      setGlobalResult(res.data)
    } catch (e) {
      setGlobalError(e.response?.data?.detail || e.message || 'Prediction failed')
    } finally { setGlobalLoading(false) }
  }

  async function runClient(file) {
    setClientFile(file); setClientLoading(true); setClientError(''); setClientResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post(
        `${API}/predict_csv_client?client_id=${selectedClient}`,
        form,
        { headers: { ...authHeader() } },  // browser sets multipart boundary automatically
      )
      setClientResult(res.data)
    } catch (e) {
      setClientError(e.response?.data?.detail || e.message || 'Prediction failed. Ensure local model is saved.')
    } finally { setClientLoading(false) }
  }

  return (
    <div>
      {/* Instructions */}
      <Panel title="How to use" tag="GUIDE" accent="#ffaa00">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {[
            ['1. Generate test CSVs', 'Run python generate_test_csvs.py from your project root. Files appear in tests/csv/.'],
            ['2. Upload & predict',   'Upload any CSV to the Global model panel OR a specific Client model panel. Label column is optional but enables accuracy/F1/confusion matrix.'],
            ['3. Compare',            'Upload the same CSV to both panels to see side-by-side comparison proving the global model generalises better.'],
          ].map(([title, desc]) => (
            <div key={title} style={{ background: 'rgba(255,170,0,0.04)', border: '1px solid rgba(255,170,0,0.12)', borderRadius: 8, padding: '12px 14px' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#ffaa00', marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>{desc}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(0,212,255,0.04)', borderRadius: 6, fontSize: 11, color: '#475569', fontFamily: 'monospace' }}>
          Test files: test_balanced.csv · test_client1_bias.csv · test_client2_bias.csv · test_client3_bias.csv · test_hard.csv · test_small.csv
        </div>
      </Panel>

      {/* Upload panels */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>

        {/* Global model */}
        <Panel title="Global Federated Model" tag="GLOBAL" accent="#00d4ff">
          <DropZone onFile={runGlobal} loading={globalLoading} label="Upload CSV → predict with global model" accent="#00d4ff" />
          {globalFile && !globalLoading && (
            <div style={{ marginTop: 8, fontSize: 11, color: '#475569', fontFamily: 'monospace' }}>
              📄 {globalFile.name}
            </div>
          )}
          {globalError && (
            <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(255,60,60,0.08)', border: '1px solid rgba(255,60,60,0.2)', borderRadius: 6, fontSize: 12, color: '#ff6b6b' }}>
              {globalError}
            </div>
          )}
        </Panel>

        {/* Client model */}
        <Panel title="Client Local Model" tag={`CLIENT ${selectedClient}`} accent={MODEL_COLORS[`client_${selectedClient}`] || '#a78bfa'}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            {[1, 2, 3].map(id => (
              <button
                key={id}
                onClick={() => { setSelectedClient(id); setClientResult(null); setClientError('') }}
                style={{
                  flex: 1, padding: '8px 0',
                  background: selectedClient === id ? `${MODEL_COLORS[`client_${id}`] || '#a78bfa'}18` : 'transparent',
                  border: `1px solid ${selectedClient === id ? (MODEL_COLORS[`client_${id}`] || '#a78bfa') + '60' : 'rgba(255,255,255,0.08)'}`,
                  borderRadius: 6,
                  color: selectedClient === id ? MODEL_COLORS[`client_${id}`] || '#a78bfa' : '#475569',
                  fontFamily: 'monospace', fontSize: 12, fontWeight: 600,
                  cursor: 'pointer', transition: 'all 0.15s',
                }}
              >
                Client {id}
              </button>
            ))}
          </div>
          <DropZone
            onFile={runClient}
            loading={clientLoading}
            label={`Upload CSV → predict with Client ${selectedClient}'s local model`}
            accent={MODEL_COLORS[`client_${selectedClient}`] || '#a78bfa'}
          />
          {clientFile && !clientLoading && (
            <div style={{ marginTop: 8, fontSize: 11, color: '#475569', fontFamily: 'monospace' }}>
              📄 {clientFile.name}
            </div>
          )}
          {clientError && (
            <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(255,60,60,0.08)', border: '1px solid rgba(255,60,60,0.2)', borderRadius: 6, fontSize: 12, color: '#ff6b6b' }}>
              {clientError}
            </div>
          )}
        </Panel>
      </div>

      {/* Comparison — only when both results exist */}
      {globalResult && clientResult && (
        <ComparePanel globalResult={globalResult} clientResult={clientResult} clientId={selectedClient} />
      )}

      {/* Individual results */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {globalResult && (
          <ResultPanel result={globalResult} modelLabel="Global Model Results" accent="#00d4ff" />
        )}
        {clientResult && (
          <ResultPanel result={clientResult} modelLabel={`Client ${selectedClient} Local Model Results`} accent={MODEL_COLORS[`client_${selectedClient}`] || '#a78bfa'} />
        )}
      </div>
    </div>
  )
}