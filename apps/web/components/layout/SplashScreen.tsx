"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";

export function SplashScreen() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Show splash for a fixed duration then fade out
    const timer = setTimeout(() => {
      setLoading(false);
    }, 2500);

    return () => clearTimeout(timer);
  }, []);

  return (
    <AnimatePresence>
      {loading && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.05, filter: "brightness(2)" }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background pointer-events-auto"
        >
          {/* Subtle background noise */}
          <div className="absolute inset-0 z-0 h-full w-full opacity-[0.03] mix-blend-overlay bg-[url('/noise.svg')]" />

          <div className="relative z-10 flex flex-col items-center">
            {/* Blinking Logo */}
            <motion.div
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{
                repeat: Infinity,
                duration: 1.5,
                ease: "easeInOut",
              }}
              className="relative size-24 md:size-32 drop-shadow-[0_0_24px_rgba(var(--color-primary),0.4)]"
            >
              <Image
                src="/favicons/logo.png"
                alt="Matcha AI"
                fill
                className="object-contain"
                priority
                sizes="(max-width: 768px) 96px, 128px"
              />
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
