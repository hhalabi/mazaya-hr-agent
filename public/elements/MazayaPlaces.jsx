// public/elements/MazayaPlaces.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";

/** Load Google Maps JS once and reuse. */
function loadGMaps(apiKey, language = "en", region = "SA") {
  if (!apiKey) return Promise.reject(new Error("Missing GOOGLE_MAPS_API_KEY"));
  if (window.google && window.google.maps) return Promise.resolve(window.google);

  if (!window._gmapsLoading) {
    window._gmapsLoading = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      // Places + Advanced Markers
      s.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places,marker&v=weekly&language=${language}&region=${region}`;
      s.async = true;
      s.onerror = () => reject(new Error("Failed to load Google Maps JS"));
      s.onload = () => resolve(window.google);
      document.head.appendChild(s);
    });
  }
  return window._gmapsLoading;
}

function cls(...xs) {
  return xs.filter(Boolean).join(" ");
}

// Minimal badge for open/closed
function OpenBadge({ opening_hours }) {
  const open = opening_hours?.isOpen?.() ?? opening_hours?.isOpen ?? null;
  if (open === null) return null;
  return (
    <span
      style={{
        fontSize: 12,
        padding: "2px 8px",
        borderRadius: 999,
        marginRight: 6,
        background: open ? "rgba(16,185,129,.15)" : "rgba(239,68,68,.15)",
        color: open ? "#10b981" : "#ef4444",
        border: `1px solid ${open ? "#10b981" : "#ef4444"}`
      }}
    >
      {open ? "Open now" : "Closed"}
    </span>
  );
}

export default function MazayaPlaces() {
  // All props come from the global `props` (Chainlit rule)
  const {
    apiKey,
    mapId,                 // <<< NEW
    queries = [],          // [{ id, title, provider, category, city, country?, url? }]
    center = null,         // {lat, lng} optional initial center
    language = "en",
    region = "SA",         // helps bias to Saudi
    max = 20,              // max places to resolve
    listFirst = true       // default to List view like your screenshot
  } = (typeof props === "object" && props) || {};

  const [view, setView] = useState(listFirst ? "list" : "map");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [places, setPlaces] = useState([]); // [{placeId, name, lat, lng, ...extra}]
  const [selected, setSelected] = useState(0);
  const mapRef = useRef(null);
  const listRef = useRef(null);
  const gmap = useRef(null);
  const markers = useRef([]);
  const bounds = useRef(null);

  const normalized = useMemo(() => {
    // Build user-friendly text search queries for Places
    return (queries || []).slice(0, max).map((q, i) => {
      const name = q.provider || q.title || "";
      const city = q.city || "";
      const country = q.country || "Saudi Arabia";
      const category = q.category || "";
      const query = [name, category, city, country].filter(Boolean).join(", ");
      return { ...q, idx: i, query };
    });
  }, [JSON.stringify(queries), max]);

  // Resolve each query -> Place → Details
  useEffect(() => {
    let cancelled = false;
    async function run() {
      setBusy(true);
      setError("");
      setPlaces([]);
      markers.current.forEach((m) => m?.setMap && m.setMap(null));
      markers.current = [];

      try {
        const google = await loadGMaps(apiKey, language, region);
        if (!mapRef.current) return;

        const initialCenter =
          center ||
          { lat: 23.8859, lng: 45.0792 }; // fallback SA center

        gmap.current = new google.maps.Map(mapRef.current, {
          center: initialCenter,
          zoom: 6,
          mapId: mapId || undefined,        // <<< NEW (required for Advanced Markers)
          disableDefaultUI: false,
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: true
        });

        bounds.current = new google.maps.LatLngBounds();
        const svc = new google.maps.places.PlacesService(gmap.current);

        // Helper: Text search → first candidate
        const textSearch = (query) =>
          new Promise((resolve) =>
            svc.textSearch(
              { query, region, fields: ["place_id", "geometry"] },
              (res, status) => {
                if (status === google.maps.places.PlacesServiceStatus.OK && res?.length) {
                  resolve(res[0]);
                } else {
                  resolve(null);
                }
              }
            )
          );

        // Helper: details
        const getDetails = (placeId) =>
          new Promise((resolve) =>
            svc.getDetails(
              {
                placeId,
                fields: [
                  "place_id",
                  "name",
                  "rating",
                  "user_ratings_total",
                  "formatted_address",
                  "geometry",
                  "opening_hours",
                  "international_phone_number",
                  "website",
                  "url",
                  "photos"
                ]
              },
              (res, status) => {
                if (status === google.maps.places.PlacesServiceStatus.OK && res) {
                  resolve(res);
                } else {
                  resolve(null);
                }
              }
            )
          );

        const resolved = [];
        // Resolve in sequence (safer for quotas)
        for (const q of normalized) {
          if (cancelled) return;
          const found = await textSearch(q.query);
          if (!found?.place_id) continue;
          const det = await getDetails(found.place_id);
          if (!det?.geometry?.location) continue;

          const lat = det.geometry.location.lat();
          const lng = det.geometry.location.lng();
          const photo =
            det.photos?.[0]?.getUrl?.({ maxWidth: 640, maxHeight: 360 }) ||
            null;

          resolved.push({
            // base
            idx: q.idx,
            provider: q.provider,
            title: q.title,
            category: q.category,
            city: q.city,
            urlHint: q.url,
            // place
            placeId: det.place_id,
            name: det.name,
            address: det.formatted_address,
            rating: det.rating,
            ratings: det.user_ratings_total,
            opening_hours: det.opening_hours || null,
            phone: det.international_phone_number || null,
            website: det.website || null,
            mapsUrl: det.url || null,
            photo,
            lat,
            lng
          });
        }

        if (cancelled) return;

        // Render markers & fit
        const canUseAdvanced = !!(google.maps.marker?.AdvancedMarkerElement && mapId); // <<< NEW gate
        resolved.forEach((p, i) => {
          const pos = { lat: p.lat, lng: p.lng };
          bounds.current.extend(pos);

          const marker = canUseAdvanced
            ? new google.maps.marker.AdvancedMarkerElement({
                map: gmap.current,
                position: pos,
                title: p.name || p.provider || p.title
              })
            : new google.maps.Marker({
                map: gmap.current,
                position: pos,
                title: p.name || p.provider || p.title,
                label: String(i + 1)
              });

          marker.addListener?.("click", () => {
            setSelected(i);
            if (listRef.current) {
              const card = listRef.current.querySelector(`[data-idx="${i}"]`);
              if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            }
          });

          markers.current.push(marker);
        });

        if (resolved.length > 1) {
          gmap.current.fitBounds(bounds.current, { top: 24, bottom: 24, left: 24, right: 24 });
        } else if (resolved.length === 1) {
          gmap.current.setCenter({ lat: resolved[0].lat, lng: resolved[0].lng });
          gmap.current.setZoom(14);
        }

        setPlaces(resolved);
        setBusy(false);
      } catch (e) {
        console.error(e);
        setError(e?.message || String(e));
        setBusy(false);
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [apiKey, mapId, JSON.stringify(normalized), JSON.stringify(center), language, region]); // <<< mapId added

  // Keep map centered on selection
  useEffect(() => {
    const p = places[selected];
    if (!p || !gmap.current) return;
    gmap.current.panTo({ lat: p.lat, lng: p.lng });
    gmap.current.setZoom(Math.max(gmap.current.getZoom(), 14));
  }, [selected, JSON.stringify(places)]);

  // UI -----------------------------------------------------------------------
  return (
    <div style={{ width: "100%", borderRadius: 12, overflow: "hidden", border: "1px solid #333", background: "#111" }}>
      {/* Header */}
      <div style={{ padding: "10px 14px", borderBottom: "1px solid #222", color: "#ddd", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ fontWeight: 600, fontSize: 16 }}>
          {places.length ? `Found ${places.length} place${places.length > 1 ? "s" : ""}` : "Searching places…"}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setView("list")}
            className={cls("cz-btn")}
            style={{
              fontSize: 12,
              padding: "6px 10px",
              borderRadius: 8,
              border: "1px solid #333",
              background: view === "list" ? "#222" : "transparent",
              color: "#eee",
              cursor: "pointer"
            }}
          >
            List
          </button>
          <button
            onClick={() => setView("map")}
            className={cls("cz-btn")}
            style={{
              fontSize: 12,
              padding: "6px 10px",
              borderRadius: 8,
              border: "1px solid #333",
              background: view === "map" ? "#222" : "transparent",
              color: "#eee",
              cursor: "pointer"
            }}
          >
            Map
          </button>
        </div>
      </div>

      {/* Map */}
      <div style={{ width: "100%", height: view === "map" ? 420 : 260, transition: "height .2s ease" }}>
        <div ref={mapRef} style={{ width: "100%", height: "100%" }} />
      </div>

      {/* List / Carousel */}
      <div
        ref={listRef}
        style={{
          display: "flex",
          gap: 12,
          overflowX: "auto",
          padding: 12,
          borderTop: "1px solid #222",
          background: "#0c0c0c"
        }}
      >
        {places.map((p, i) => (
          <div
            key={p.placeId || p.idx}
            data-idx={i}
            onClick={() => setSelected(i)}
            style={{
              minWidth: 320,
              maxWidth: 360,
              background: i === selected ? "#161616" : "#121212",
              border: i === selected ? "1px solid #4b5563" : "1px solid #222",
              borderRadius: 12,
              overflow: "hidden",
              cursor: "pointer",
            }}
          >
            <div style={{ width: "100%", height: 160, background: "#1a1a1a" }}>
              {p.photo ? (
                <img src={p.photo} alt={p.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              ) : null}
            </div>
            <div style={{ padding: 12, color: "#ddd" }}>
              <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 4 }}>
                {i + 1}. {p.name || p.provider || p.title}
              </div>
              <div style={{ fontSize: 12, color: "#aaa" }}>
                {p.provider && <span>{p.provider}</span>}
                {p.provider && p.category && <span> · </span>}
                {p.category && <span>{p.category}</span>}
              </div>

              <div style={{ marginTop: 6, fontSize: 12, color: "#bbb" }}>{p.address}</div>

              <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 8 }}>
                {typeof p.rating === "number" && (
                  <span style={{ fontSize: 12 }}>
                    ⭐ {p.rating.toFixed(1)} {p.ratings ? `(${p.ratings} reviews)` : ""}
                  </span>
                )}
                <OpenBadge opening_hours={p.opening_hours} />
              </div>

              <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
                <a
                  href={`https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(p.name)}&destination_place_id=${p.placeId}`}
                  target="_blank"
                  rel="noopener"
                  style={btnStyle()}
                >
                  Directions
                </a>
                {p.website && (
                  <a href={p.website} target="_blank" rel="noopener" style={btnStyle()}>
                    Website
                  </a>
                )}
                {p.phone && (
                  <a href={`tel:${p.phone.replace(/\s+/g, "")}`} style={btnStyle()}>
                    Call
                  </a>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Footer / states */}
      {(busy || error) && (
        <div style={{ padding: 10, borderTop: "1px solid #222", color: error ? "#ef4444" : "#999" }}>
          {error ? `Map error: ${error}` : "Fetching details…"}
        </div>
      )}
      {!places.length && !busy && !error && (
        <div style={{ padding: 10, borderTop: "1px solid #222", color: "#bbb" }}>No places found.</div>
      )}
    </div>
  );
}

function btnStyle() {
  return {
    fontSize: 12,
    padding: "6px 10px",
    borderRadius: 8,
    border: "1px solid #333",
    color: "#eee",
    background: "#1a1a1a",
    textDecoration: "none"
  };
}
