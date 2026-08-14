import { Bell, ChevronDown } from 'lucide-react'

export function Topbar() {
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border px-4 md:px-6">
      <div className="flex flex-col leading-tight">
        <h1 className="text-base font-semibold tracking-tight">
          Video Audit Dashboard
        </h1>
        <p className="font-mono text-[11px] text-muted-foreground">
          Case #4471-B · North Terminal Cameras
        </p>
      </div>

      <div className="flex items-center gap-4">
        <button
          type="button"
          className="relative flex size-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          aria-label="Notifications"
        >
          <Bell className="size-4" aria-hidden="true" />
          <span className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-primary" />
        </button>
        <button
          type="button"
          className="flex items-center gap-2 rounded-md py-1.5 pl-1.5 pr-2 transition-colors hover:bg-secondary"
        >
          <div className="flex size-7 items-center justify-center rounded-full bg-primary/15 text-xs font-semibold text-primary ring-1 ring-primary/30">
            AR
          </div>
          <span className="hidden text-sm font-medium sm:inline">
            A. Ramirez
          </span>
          <ChevronDown
            className="size-3.5 text-muted-foreground"
            aria-hidden="true"
          />
        </button>
      </div>
    </header>
  )
}
