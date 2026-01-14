import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import QueryProvider from "@/src/providers/query-provider";
import { GlobalProvider } from "../context/GlobalContext";
import { ThemeProvider } from "../components/ThemeProvider";
import { SidebarProvider } from "../context/SidebarContext";
import Sidebar from "../components/layout/Sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Nepali Sign Language",
  description: "Learn and practice Nepali Sign Language",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-screen overflow-hidden`}
      >
        
        <QueryProvider>
          <GlobalProvider>
            <ThemeProvider
              attribute="class"
              defaultTheme="system"
              enableSystem
            >
              <SidebarProvider>
                <div className="flex min-h-screen bg-secondary/10">
                  {/* The Sidebar component manages its own responsive width */}
                  <Sidebar />
                  
                  {/* Main Content Area */}
                  <main className="flex-1 p-3 md:p-4 h-screen overflow-hidden">
                    <div className="max-w-360 h-full mx-auto overflow-y-scroll scrollbar-hide">
                        {children}
                    </div>
                  </main>
                </div>
              </SidebarProvider>
            </ThemeProvider>
          </GlobalProvider>
        </QueryProvider>
      </body>
    </html>
  );
}