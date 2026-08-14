'use client'

import { useState } from 'react'
import { LayoutDashboard, History, Shield, Settings, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { label: 'Dashboard', icon: LayoutDashboard, href: '#', active: true },
  { label: 'Audit History', icon: History, href: '#' },
  { label: 'Evidence Locker', icon: Shield, href: '#' },
  { label: 'System Settings', icon: Settings, href: '#' },
]

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        'hidden md:flex flex-col border-r border-border bg-card transition-all duration-300 relative select-none',
        isCollapsed ? 'w-20' : 'w-64'
      )}
    >
      {/* Brand Header */}
      <div className="flex h-16 items-center gap-3 px-6 border-b border-border overflow-hidden">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 border border-primary/20 text-primary">
          <Shield className="size-4" />
        </div>
        {!isCollapsed && (
          <div className="flex flex-col truncate">
            <span className="font-bold tracking-wider text-sm">SENTINEL</span>
            <span className="text-[10px] tracking-widest text-muted-foreground uppercase font-mono">
              Audit Console
            </span>
          </div>
        )}
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 space-y-1.5 p-3">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <a
              key={item.label}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3.5 py-2.5 text-sm font-medium transition-colors',
                item.active
                  ? 'bg-primary/10 text-primary border border-primary/20'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
              )}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon className="size-4 shrink-0" />
              {!isCollapsed && <span className="truncate">{item.label}</span>}
            </a>
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