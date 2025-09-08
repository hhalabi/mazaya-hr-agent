// public/elements/MazayaMap.jsx
import React, { useEffect, useRef } from "react";

/** Load Google Maps JS once and reuse. */
function loadGMaps(apiKey) {
  if (!apiKey) return Promise.reject(new Error("Missing GOOGLE_MAPS_API_KEY"));
  if (window.google && window.google.maps) return Promise.resolve(window.google);

  if (!window._gmapsLoading) {
    window._gmapsLoading = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places&v=quarterly`;
      s.async = true;
      s.onerror = () => reject(new Error("Failed to load Google Maps JS"));
      s.onload = () => resolve(window.google);
      document.head.appendChild(s);
    });
  }
  return window._gmapsLoading;
}

export default function MazayaMap() {
  // ⬇️ Pull everything from the globally-injected `props` object
  const {
    apiKey,
    markers = [],
    fitBounds = true,
    zoom = 6,
    height = 420
  } = (typeof props === "object" && props) || {};

  const mapRef = useRef(null);

  useEffect(() => {
    let map, bounds;

    loadGMaps(apiKey)
      .then((google) => {
        if (!mapRef.current) return;

        const fallbackKSA = { lat: 23.8859, lng: 45.0792 }; // Saudi Arabia
        const first =
          markers.length > 0
            ? {
                lat: Number(markers[0].lat) || 24.7136,
                lng: Number(markers[0].lng) || 46.6753
              }
            : fallbackKSA;

        map = new google.maps.Map(mapRef.current, {
          center: first,
          zoom,
          disableDefaultUI: false,
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: true
        });

        bounds = new google.maps.LatLngBounds();

        markers.forEach((m) => {
          const pos = { lat: Number(m.lat), lng: Number(m.lng) };
          if (Number.isNaN(pos.lat) || Number.isNaN(pos.lng)) return;

          const marker = new google.maps.Marker({
            position: pos,
            map,
            title: m.title || m.provider
          });

          const html = `
            <div style="font-family: system-ui; max-width: 240px;">
              <div style="font-weight:600;margin-bottom:4px">${m.title || ""}</div>
              <div style="font-size:12px;color:#555">${m.provider || ""} — ${m.category || ""}</div>
              <div style="font-size:12px;margin-top:6px">${(m.city || "").toUpperCase()}</div>
              ${
                m.url
                  ? `<a href="${m.url}" target="_blank" rel="noopener" style="font-size:12px;margin-top:6px;display:inline-block">Open offer</a>`
                  : ""
              }
            </div>
          `;
          const infow = new google.maps.InfoWindow({ content: html });
          marker.addListener("click", () => infow.open({ anchor: marker, map }));

          if (fitBounds) bounds.extend(pos);
        });

        if (fitBounds && markers.length > 1) {
          map.fitBounds(bounds, { top: 30, bottom: 30, left: 30, right: 30 });
        }
      })
      .catch((e) => {
        if (mapRef.current) {
          mapRef.current.innerHTML = `<div style="padding:12px;color:#c00">Map error: ${e.message}</div>`;
        }
      });
  // Re-render when the props we care about change
  }, [apiKey, JSON.stringify(markers), fitBounds, zoom, height]);

  return (
    <div
      style={{
        width: "100%",
        height,
        minHeight: height,
        position: "relative",
        borderRadius: 12,
        overflow: "hidden",
        border: "1px solid #333",
        display: "block"
      }}
    >
      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />
      {!apiKey && (
        <div style={{ padding: 12, color: "#c00" }}>
          Missing GOOGLE_MAPS_API_KEY
        </div>
      )}
      {(markers?.length || 0) === 0 && (
        <div
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            background: "rgba(0,0,0,0.6)",
            color: "#fff",
            padding: "4px 8px",
            borderRadius: 8
          }}
        >
          No locations to display
        </div>
      )}
    </div>
  );
}
