"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";

const STORAGE_KEY     = "matcha_admin_mode";
const AD_STORAGE_KEY  = "matcha_ad_overlay_url";

interface AdminContextValue {
  isAdmin: boolean;
  toggleAdmin: () => void;
  enableAdmin: () => void;
  disableAdmin: () => void;
  /** Data-URL or remote URL of the ad/sponsor image to overlay on all videos */
  adOverlayUrl: string | null;
  setAdOverlayUrl: (url: string | null) => void;
}

const AdminContext = createContext<AdminContextValue>({
  isAdmin: false,
  toggleAdmin: () => {},
  enableAdmin: () => {},
  disableAdmin: () => {},
  adOverlayUrl: null,
  setAdOverlayUrl: () => {},
});

export function AdminProvider({ children }: { children: React.ReactNode }) {
  const [isAdmin, setIsAdmin] = useState(false);
  const [adOverlayUrl, _setAdOverlayUrl] = useState<string | null>(null);

  // Hydrate from localStorage on mount
  useEffect(() => {
    try {
      setIsAdmin(localStorage.getItem(STORAGE_KEY) === "true");
      const saved = localStorage.getItem(AD_STORAGE_KEY);
      if (saved) _setAdOverlayUrl(saved);
    } catch {}
  }, []);

  const setAdOverlayUrl = useCallback((url: string | null) => {
    _setAdOverlayUrl(url);
    try {
      if (url) localStorage.setItem(AD_STORAGE_KEY, url);
      else localStorage.removeItem(AD_STORAGE_KEY);
    } catch {}
  }, []);

  const toggleAdmin = useCallback(() => {
    setIsAdmin(prev => {
      const next = !prev;
      try { localStorage.setItem(STORAGE_KEY, String(next)); } catch {}
      return next;
    });
  }, []);

  const enableAdmin = useCallback(() => {
    setIsAdmin(true);
    try { localStorage.setItem(STORAGE_KEY, "true"); } catch {}
  }, []);

  const disableAdmin = useCallback(() => {
    setIsAdmin(false);
    try { localStorage.setItem(STORAGE_KEY, "false"); } catch {}
  }, []);

  return (
    <AdminContext.Provider value={{ isAdmin, toggleAdmin, enableAdmin, disableAdmin, adOverlayUrl, setAdOverlayUrl }}>
      {children}
    </AdminContext.Provider>
  );
}

export function useAdmin() {
  return useContext(AdminContext);
}
