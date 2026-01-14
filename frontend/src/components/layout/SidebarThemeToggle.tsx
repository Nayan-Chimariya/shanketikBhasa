'use client';

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function SidebarThemeToggle({ isCollapsed }: { isCollapsed: boolean }) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null; // Avoid hydration mismatch

  // --- COLLAPSED STATE (Simple Icon Button) ---
  if (isCollapsed) {
    return (
      <button
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="w-10 h-10 mx-auto mb-4 bg-secondary/50 hover:bg-secondary rounded-full flex items-center justify-center transition-colors text-muted-foreground hover:text-foreground"
      >
        {theme === 'dark' ? <Moon size={18} /> : <Sun size={18} />}
      </button>
    );
  }

  // --- EXPANDED STATE (Sliding Pill Design) ---
  return (
    <div className="bg-secondary/40 p-1 rounded-full flex items-center relative mb-4">
      {/* Sliding Background */}
      <motion.div
        layout
        transition={{ type: "spring", stiffness: 400, damping: 30 }}
        className={cn(
          "absolute top-1 bottom-1 w-[calc(50%-4px)] bg-background rounded-full shadow-sm z-0",
          theme === "dark" ? "left-[calc(50%+2px)]" : "left-1"
        )}
      />

      {/* Light Button */}
      <button
        onClick={() => setTheme("light")}
        className={cn(
          "flex-1 flex items-center justify-center gap-2 py-1.5 text-sm font-medium z-10 transition-colors rounded-full",
          theme === "light" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        )}
      >
        <Sun size={16} />
        <span>Light</span>
      </button>

      {/* Dark Button */}
      <button
        onClick={() => setTheme("dark")}
        className={cn(
          "flex-1 flex items-center justify-center gap-2 py-1.5 text-sm font-medium z-10 transition-colors rounded-full",
          theme === "dark" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        )}
      >
        <Moon size={16} />
        <span>Dark</span>
      </button>
    </div>
  );
}