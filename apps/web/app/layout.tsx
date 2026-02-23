import type { Metadata } from "next";
import { Bebas_Neue, Barlow_Condensed, DM_Mono } from "next/font/google";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { SplashScreen } from "@/components/layout/SplashScreen";
import { AuthProvider } from "@/contexts/AuthContext";
import { env } from "@/lib/env";
import "./globals.css";

const bebas = Bebas_Neue({
  variable: "--font-bebas",
  subsets: ["latin"],
  weight: "400",
});

const barlow = Barlow_Condensed({
  variable: "--font-barlow",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  style: ["normal", "italic"],
});

const dmMono = DM_Mono({
  variable: "--font-dm-mono",
  subsets: ["latin"],
  weight: ["300", "400", "500"],
});

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
      <head />
      <body suppressHydrationWarning className={`${bebas.variable} ${barlow.variable} ${dmMono.variable} antialiased min-h-screen flex flex-col bg-background text-foreground selection:bg-primary/30 selection:text-primary`}>
        <AuthProvider>
          <SplashScreen />
          <div className="flex-1 flex flex-col relative w-full">
            <Navbar />
            <main className="flex-1 flex flex-col items-stretch w-full relative z-10">
              {children}
            </main>
            <Footer />
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
