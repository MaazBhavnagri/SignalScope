import { useCallback, useRef, useState } from 'react'

interface Props { onFiles: (files: File[]) => void; multiple?: boolean; title: string; hint: string; compact?: boolean; disabled?: boolean }

export default function Dropzone({ onFiles, multiple = false, title, hint, compact = false, disabled = false }: Props) {
  const [over, setOver] = useState(false)
  const ref = useRef<HTMLInputElement>(null)
  const accept = useCallback((list: FileList | null) => {
    if (!list) return
    const files = Array.from(list).filter(f => f.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp|tiff?|gif)$/i.test(f.name))
    if (files.length) onFiles(multiple ? files : [files[0]])
  }, [onFiles, multiple])
  return (
    <div className={`drop ${over ? 'over' : ''}`} style={compact ? { minHeight: 140 } : undefined}
      onDragOver={e => { e.preventDefault(); if (!disabled) setOver(true) }} onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false); if (!disabled) accept(e.dataTransfer.files) }}
      onPaste={e => { if (!disabled) accept(e.clipboardData.files) }} tabIndex={0} role="button" aria-label={title}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') ref.current?.click() }}>
      <span className="reticle tl" /><span className="reticle tr" /><span className="reticle bl" /><span className="reticle br" />
      <input ref={ref} type="file" accept="image/*" multiple={multiple} disabled={disabled} onChange={e => { accept(e.target.files); e.target.value = '' }} />
      <div>
        {!compact && (
          <svg className="icon" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth={1.6}>
            <circle cx="24" cy="24" r="14" /><path d="M24 4v8M24 36v8M4 24h8M36 24h8" strokeLinecap="round" /><circle cx="24" cy="24" r="4" fill="currentColor" />
          </svg>
        )}
        <h3 style={compact ? { fontSize: 18 } : undefined}>{title}</h3>
        <p>{hint}</p>
      </div>
    </div>
  )
}
