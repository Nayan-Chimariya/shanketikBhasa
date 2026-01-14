'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSidebar } from '@/src/context/SidebarContext';
import { cn } from '@/lib/utils';
import { 
  LayoutDashboard, Calendar, FileText, Inbox, 
  BarChart2, MessageSquare, Info, HelpCircle, 
  Search, ChevronRight, ChevronLeft, MoreVertical,
  LogOut, Settings, ChevronDown, Folder, User as UserIcon,
  House, GraduationCap, NotebookPen
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { SidebarThemeToggle } from './SidebarThemeToggle'; 
import Image from 'next/image';
import logo from '@/src/assets/images/logowhite.png';

// --- MOCK NAV DATA ---
const NAV_ITEMS = [
  { label: 'Menu', isHeader: true },
  { icon: House, label: 'Home', href: '/' },
  { icon: LayoutDashboard, label: 'Dashboard', href: '/dashboard' },
  { icon: GraduationCap, label: 'Learn', href: '/learn' },
  { icon: NotebookPen, label: 'Test', href: '/test' },
  { 
    icon: Folder, 
    label: 'Documents', 
    isGroup: true,
    children: [
      { icon: Inbox, label: 'Inbox', href: '/documents/inbox' },
      { icon: FileText, label: 'Incoming', href: '/documents/incoming' },
      { icon: FileText, label: 'Export', href: '/documents/export' },
    ]
  },
  { icon: BarChart2, label: 'Statistics', href: '/statistics' },
  { icon: MessageSquare, label: 'Chat', href: '/chat', badge: 5 },
  
  { label: 'Others', isHeader: true },
  { icon: Info, label: 'Info', href: '/info' },
  { icon: HelpCircle, label: 'Request', href: '/request' },
];

export default function Sidebar() {
  const { isCollapsed, toggleSidebar } = useSidebar();
  
  return (
    <motion.aside
      initial={false}
      animate={{ width: isCollapsed ? 80 : 280 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      // Added 'overflow-visible' so popups can escape the container bounds
      className="h-screen bg-background border-r border-border flex flex-col sticky top-0 left-0 z-40 shadow-sm overflow-visible"
    >
      {/* --- HEADER --- */}
      <div className="h-20 flex items-center justify-between px-2 shrink-0 relative z-50">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="min-w-14 h-14 overflow-hidden rounded-full flex items-center justify-center text-primary-foreground font-bold text-lg">
            <Image src={logo} alt="Logo" width={52} height={52} className="object-cover" />
          </div>
          <motion.span 
            animate={{ opacity: isCollapsed ? 0 : 1, width: isCollapsed ? 0 : 'auto' }}
            className="font-bold text-xl whitespace-nowrap overflow-hidden text-black dark:text-white"
          >
            𝐒𝐚𝐧𝐤𝐞𝐭𝐢𝐜 𝐁𝐡𝐚𝐬𝐚
          </motion.span>
        </div>
        
        <button 
          onClick={toggleSidebar}
          className="p-1.5 rounded-md hover:bg-muted text-muted-foreground transition-colors absolute -right-3 top-7 bg-background border border-border shadow-sm z-50"
        >
          {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {/* --- SEARCH --- */}
      <div className="px-4 mb-6 shrink-0 relative z-40">
        <div className={cn(
          "flex items-center bg-secondary/50 rounded-xl px-3 h-10 transition-all",
          isCollapsed ? "justify-center cursor-pointer" : ""
        )}>
          <Search size={18} className="text-muted-foreground shrink-0" />
          {!isCollapsed && (
            <input 
              type="text" 
              placeholder="Search" 
              className="bg-transparent border-none outline-none text-sm ml-3 w-full text-foreground placeholder:text-muted-foreground"
            />
          )}
          {!isCollapsed && <Settings size={14} className="text-muted-foreground" /> }
        </div>
      </div>

      {/* --- NAVIGATION --- */}
      {/* 
         CRITICAL FIX: 
         When Collapsed: overflow-visible (allows tooltips/popups to show outside).
         When Expanded: overflow-y-auto (allows scrolling).
      */}
      <div className={cn(
        "flex-1 px-4 space-y-1 pb-4",
        isCollapsed ? "overflow-visible" : "better-scroll overflow-y-auto overflow-x-hidden"
      )}>
        {NAV_ITEMS.map((item, idx) => (
          <React.Fragment key={idx}>
            {item.isHeader ? (
              !isCollapsed && (
                <div className="text-xs font-medium text-muted-foreground mt-6 mb-2 px-2 uppercase tracking-wider">
                  {item.label}
                </div>
              )
            ) : item.isGroup ? (
              <SidebarGroup item={item} isCollapsed={isCollapsed} />
            ) : (
              <SidebarItem item={item} isCollapsed={isCollapsed} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* --- FOOTER --- */}
      <div className="p-4 border-t border-border mt-auto shrink-0 relative z-50 bg-background">
        <SidebarThemeToggle isCollapsed={isCollapsed} />
        <UserProfile isCollapsed={isCollapsed} />
      </div>
    </motion.aside>
  );
}

// --- SUB COMPONENTS ---

function SidebarItem({ item, isCollapsed }: { item: any, isCollapsed: boolean }) {
  const pathname = usePathname();
  const isActive = pathname === item.href;

  return (
    <Link href={item.href || '#'} className="group relative flex items-center z-40">
      <div className={cn(
        "flex items-center w-full p-2.5 rounded-xl transition-all duration-200",
        isActive 
          ? "bg-primary text-primary-foreground shadow-md shadow-primary/20" 
          : "text-muted-foreground hover:bg-secondary hover:text-foreground",
        isCollapsed ? "justify-center" : "justify-between"
      )}>
        <div className="flex items-center gap-3">
          <item.icon size={20} className={cn("shrink-0", isActive ? "text-primary-foreground" : "")} />
          {!isCollapsed && <span className="text-sm font-medium whitespace-nowrap">{item.label}</span>}
        </div>

        {!isCollapsed && item.badge && (
          <span className="bg-red-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full min-w-4.5 flex justify-center">
            {item.badge}
          </span>
        )}

        {/* Collapsed Tooltip - Now visible because parent overflow is visible */}
        {isCollapsed && (
          <div className="absolute left-full top-1/2 -translate-y-1/2 ml-4 px-3 py-2 bg-popover text-popover-foreground text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-[60] whitespace-nowrap border border-border">
            {item.label}
          </div>
        )}
      </div>
    </Link>
  );
}

function SidebarGroup({ item, isCollapsed }: { item: any, isCollapsed: boolean }) {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();
  const hasActiveChild = item.children.some((child: any) => pathname === child.href);

  React.useEffect(() => {
    if (hasActiveChild) setIsOpen(true);
  }, [hasActiveChild]);

  if (isCollapsed) {
    return (
      <div className="group relative z-40">
        <button className={cn(
          "flex items-center justify-center w-full p-2.5 rounded-xl transition-all duration-200 text-muted-foreground hover:bg-secondary hover:text-foreground",
          hasActiveChild && "text-primary bg-primary/10"
        )}>
          <item.icon size={20} />
        </button>

        {/* Floating Menu for Collapsed State - Visible now */}
        <div className="absolute left-full top-0 ml-4 w-48 bg-background border border-border shadow-xl rounded-xl p-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-[60]">
          <div className="px-2 py-1.5 text-xs font-bold text-muted-foreground uppercase border-b border-border mb-1">
            {item.label}
          </div>
          {item.children.map((child: any, idx: number) => (
             <Link 
               key={idx} 
               href={child.href}
               className={cn(
                 "flex items-center gap-2 px-2 py-2 rounded-lg text-sm transition-colors",
                 pathname === child.href ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-secondary hover:text-foreground"
               )}
             >
               <child.icon size={16} />
               {child.label}
             </Link>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mb-1">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center justify-between w-full p-2.5 rounded-xl transition-all duration-200 text-muted-foreground hover:bg-secondary/50 hover:text-foreground",
          isOpen && "text-foreground"
        )}
      >
        <div className="flex items-center gap-3">
          <item.icon size={20} className={cn(hasActiveChild ? "text-primary" : "")} />
          <span className="text-sm font-medium">{item.label}</span>
        </div>
        <ChevronDown size={16} className={cn("transition-transform duration-200", isOpen ? "rotate-180" : "")} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pl-4 mt-1 space-y-1 relative">
              <div className="absolute left-5.25 top-0 bottom-2 w-px bg-border" />
              {item.children.map((child: any, idx: number) => (
                   <Link key={idx} href={child.href} className="block relative">
                     <div className={cn(
                       "flex items-center gap-3 p-2 rounded-lg ml-3 text-sm transition-all",
                       pathname === child.href
                         ? "bg-primary text-primary-foreground font-medium shadow-sm" 
                         : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                       )}>
                       <child.icon size={18} className={pathname === child.href ? "opacity-100" : "opacity-70"} />
                       {child.label}
                     </div>
                   </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function UserProfile({ isCollapsed }: { isCollapsed: boolean }) {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={menuRef}>
      <AnimatePresence>
        {showMenu && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.1 }}
            className={cn(
              "absolute bg-background border border-border rounded-xl shadow-2xl overflow-hidden z-[70] p-2 min-w-55",
              // Use left-full for collapsed mode to avoid clipping and overlap
              isCollapsed ? "left-full bottom-0 ml-4" : "bottom-full left-0 w-full mb-3"
            )}
          >
            <div className="flex items-center gap-3 p-2 mb-2 bg-secondary/30 rounded-lg">
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                AR
              </div>
              <div className="overflow-hidden">
                <p className="text-sm font-semibold truncate text-foreground">Abdullaev Rustam</p>
                <p className="text-xs text-muted-foreground truncate">rustam@gmail.com</p>
              </div>
            </div>
            
            <div className="h-px bg-border mb-2" />

            <button className="flex items-center gap-2 w-full px-3 py-2 text-sm text-foreground hover:bg-secondary rounded-lg transition-colors">
              <UserIcon size={16} />
              Profile
            </button>
            <button className="flex items-center gap-2 w-full px-3 py-2 text-sm text-foreground hover:bg-secondary rounded-lg transition-colors">
              <Settings size={16} />
              Settings
            </button>
            <div className="h-px bg-border my-1" />
            <button className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-500 hover:bg-red-50 dark:hover:bg-red-900/10 rounded-lg transition-colors">
              <LogOut size={16} />
              Logout
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <div 
        onClick={() => setShowMenu(!showMenu)}
        className={cn(
          "flex items-center gap-3 p-2 rounded-xl cursor-pointer hover:bg-secondary transition-colors relative z-10",
          isCollapsed ? "justify-center" : "",
          showMenu ? "bg-secondary" : ""
        )}
      >
        <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-gray-700 to-gray-900 overflow-hidden shrink-0 border border-border">
           <div className="w-full h-full flex items-center justify-center bg-primary text-white font-bold">
             AR
           </div>
        </div>
        
        {!isCollapsed && (
          <div className="flex-1 overflow-hidden text-left">
            <h4 className="text-sm font-semibold text-foreground truncate leading-none mb-0.5">Abdullaev Rustam</h4>
            <span className="text-xs text-muted-foreground truncate block">rustam@gmail.com</span>
          </div>
        )}

        {!isCollapsed && (
          <MoreVertical size={16} className="text-muted-foreground" />
        )}
      </div>
    </div>
  );
}