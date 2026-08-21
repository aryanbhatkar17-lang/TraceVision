'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { LayoutDashboard, History, Shield, Settings, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { label: 'Dashboard', icon: LayoutDashboard, href: '/dashboard' },
  { label: 'Audit History', icon: History, href: '/dashboard/history' },
  { label: 'Evidence Locker', icon: Shield, href: '/dashboard/evidence' },
  { label: 'System Settings', icon: Settings, href: '/dashboard/settings' },
]

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const pathname = usePathname()

  return (
    <aside
      className={cn(
        'hidden md:flex flex-col border-r border-border bg-card transition-all duration-300 relative select-none',
        isCollapsed ? 'w-20' : 'w-64'
      )}
    >
      {/* Brand Header - Links back to Landing Page */}
      <div className="flex h-16 items-center px-4 border-b border-border overflow-hidden">
        <Link
          href="/"
          className="flex items-center gap-3 w-full rounded-lg p-1.5 hover:bg-secondary/60 transition-colors"
          title="Return to Home"
        >
          <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 border border-primary/20 text-primary">
            <Shield className="size-4" />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col truncate">
              <span className="font-bold tracking-wider text-sm leading-none">TraceVision</span>
              <span className="text-[10px] tracking-widest text-muted-foreground uppercase font-mono mt-1">
                Audit Console
              </span>
            </div>
          )}
        </Link>
      </div>

      {/* Navigation Links with Next.js Client Routing */}
      <nav className="flex-1 space-y-1.5 p-3">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href

          return (
            <Link
              key={item.label}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3.5 py-2.5 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary border border-primary/20 font-semibold'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground border border-transparent'
              )}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon className="size-4 shrink-0" />
              {!isCollapsed && <span className="truncate">{item.label}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Collapse Toggle Footer */}
      <div className="p-3 border-t border-border">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={cn(
            'flex w-full items-center gap-3 rounded-lg px-3.5 py-2.5 text-sm font-medium text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors',
            isCollapsed && 'justify-center px-0'
          )}
          title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {isCollapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
          {!isCollapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  )
}