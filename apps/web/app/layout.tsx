import type { Metadata } from "next";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { SplashScreen } from "@/components/layout/SplashScreen";
import { AuthProvider } from "@/contexts/AuthContext";
import { AdminProvider } from "@/contexts/AdminContext";
import { env } from "@/lib/env";
import "./globals.css";

export const metadata: Metadata = {
  title: "Matcha AI — Match Intelligence",
  description: "AI-powered sports event detection, highlights & commentary",
  icons: {
    icon: [
      { url: "/favicons/favicon-32x32.png", sizes: "32x32" },
      { url: "/favicons/favicon-16x16.png", sizes: "16x16" },
    ],
    apple: "/favicons/apple-touch-icon.png",
  },
  manifest: "/favicons/site.webmanifest",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Google Fonts — loaded via browser link tags (no build-time download) */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=Bebas+Neue&family=Barlow+Condensed:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600;1,700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body suppressHydrationWarning className="antialiased min-h-screen flex flex-col bg-background text-foreground selection:bg-primary/30 selection:text-primary">
        <AuthProvider>
          <AdminProvider>
            <SplashScreen />
            <div className="flex-1 flex flex-col relative w-full">
              <Navbar />
              <main className="flex-1 flex flex-col items-stretch w-full relative z-10">
                {children}
              </main>
              <Footer />
            </div>
          </AdminProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
