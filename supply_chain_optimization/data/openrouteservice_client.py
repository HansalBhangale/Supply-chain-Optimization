"""
OpenRouteService API integration for real-world distances and travel times.

FREE alternative to Google Maps - uses OpenStreetMap data.
Get your free API key at: https://openrouteservice.org/dev/#/signup

Features:
- 2,000 free requests per day
- No billing required
- Real road distances and travel times
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time

try:
    import openrouteservice
    ORS_AVAILABLE = True
except ImportError:
    ORS_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Location:
    """A location with name and coordinates."""
    name: str
    lat: float
    lng: float
    address: str = ""
    
    def to_coords(self) -> List[float]:
        """Get [lng, lat] coordinates for ORS API."""
        return [self.lng, self.lat]


# Sample Indian cities for testing
SAMPLE_INDIAN_CITIES = {
    "suppliers": [
        Location("Supplier_Chennai", 13.0827, 80.2707, "Chennai, Tamil Nadu, India"),
        Location("Supplier_Mumbai", 19.0760, 72.8777, "Mumbai, Maharashtra, India"),
        Location("Supplier_Ahmedabad", 23.0225, 72.5714, "Ahmedabad, Gujarat, India"),
        Location("Supplier_Kolkata", 22.5726, 88.3639, "Kolkata, West Bengal, India"),
        Location("Supplier_Hyderabad", 17.3850, 78.4867, "Hyderabad, Telangana, India"),
    ],
    "warehouses": [
        Location("Warehouse_Bangalore", 12.9716, 77.5946, "Bangalore, Karnataka, India"),
        Location("Warehouse_Delhi", 28.6139, 77.2090, "New Delhi, Delhi, India"),
        Location("Warehouse_Pune", 18.5204, 73.8567, "Pune, Maharashtra, India"),
    ],
    "customers": [
        Location("Factory_Nagpur", 21.1458, 79.0882, "Nagpur, Maharashtra, India"),
        Location("Vendor_Jaipur", 26.9124, 75.7873, "Jaipur, Rajasthan, India"),
        Location("Vendor_Lucknow", 26.8467, 80.9462, "Lucknow, Uttar Pradesh, India"),
        Location("Vendor_Indore", 22.7196, 75.8577, "Indore, Madhya Pradesh, India"),
        Location("Vendor_Coimbatore", 11.0168, 76.9558, "Coimbatore, Tamil Nadu, India"),
        Location("Vendor_Bhopal", 23.2599, 77.4126, "Bhopal, Madhya Pradesh, India"),
        Location("Vendor_Visakhapatnam", 17.6868, 83.2185, "Visakhapatnam, Andhra Pradesh, India"),
        Location("Vendor_Surat", 21.1702, 72.8311, "Surat, Gujarat, India"),
        Location("Vendor_Kochi", 9.9312, 76.2673, "Kochi, Kerala, India"),
        Location("Vendor_Chandigarh", 30.7333, 76.7794, "Chandigarh, India"),
    ]
}


def geocode_address(address: str, api_key: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """
    Convert an address to coordinates using OpenRouteService Geocoding API.
    
    Args:
        address: Address string (e.g., "Mumbai, India")
        api_key: ORS API key (reads from ORS_API_KEY env var if not provided)
        
    Returns:
        Tuple of (lat, lng) or None if geocoding fails
    """
    import requests
    
    key = api_key or os.getenv("ORS_API_KEY")
    if not key:
        print("No ORS API key found for geocoding")
        return None
    
    try:
        url = "https://api.openrouteservice.org/geocode/search"
        params = {
            "api_key": key,
            "text": address,
            "size": 1,
            "boundary.country": "IN"  # Prioritize India
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("features") and len(data["features"]) > 0:
            coords = data["features"][0]["geometry"]["coordinates"]
            # ORS returns [lng, lat], we need (lat, lng)
            return (coords[1], coords[0])
        
        return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None


@dataclass
class ORSLocation:
    """Location with coordinates for ORS API."""
    name: str
    lng: float  # ORS uses [lng, lat] order!
    lat: float
    
    def to_coords(self) -> List[float]:
        """Get [lng, lat] coordinates for ORS API."""
        return [self.lng, self.lat]


@dataclass
class DistanceResult:
    """Result from distance matrix API."""
    distance_km: float
    duration_hours: float
    origin: str
    destination: str


class OpenRouteServiceClient:
    """
    OpenRouteService API client for distance and travel time calculations.
    
    FREE: 2,000 requests/day with free API key.
    Get your key at: https://openrouteservice.org/dev/#/signup
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize ORS client.
        
        Args:
            api_key: ORS API key (reads from ORS_API_KEY env var if not provided)
            cache_dir: Directory to cache API results
        """
        self.api_key = api_key or os.getenv("ORS_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouteService API key not found.\n"
                "1. Get free key at: https://openrouteservice.org/dev/#/signup\n"
                "2. Add to .env file: ORS_API_KEY=your_key_here"
            )
        
        if not ORS_AVAILABLE:
            raise ImportError("openrouteservice package not installed. Run: uv add openrouteservice")
        
        self.client = openrouteservice.Client(key=self.api_key)
        
        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / "cache" / "ors"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, Dict] = {}
        self._load_cache()
    
    def _get_cache_key(self, locations: List[ORSLocation]) -> str:
        """Generate cache key for a set of locations."""
        coords_str = "|".join(f"{loc.lat},{loc.lng}" for loc in locations)
        return hashlib.md5(coords_str.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cached results from disk."""
        cache_file = self.cache_dir / "distance_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "distance_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self._cache, f, indent=2)
    
    def compute_distance_matrix(
        self,
        origins: List[ORSLocation],
        destinations: List[ORSLocation],
        profile: str = "driving-car"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance and time matrices using ORS Matrix API.
        
        Args:
            origins: List of origin locations
            destinations: List of destination locations
            profile: Routing profile ('driving-car', 'driving-hgv' for trucks)
            
        Returns:
            Tuple of (distance_matrix_km, time_matrix_hours)
        """
        # Combine all locations (origins first, then destinations)
        all_locations = origins + destinations
        
        # Check cache
        cache_key = self._get_cache_key(all_locations)
        if cache_key in self._cache:
            print("  Using cached ORS distances...")
            cached = self._cache[cache_key]
            return np.array(cached["distances"]), np.array(cached["durations"])
        
        # Prepare coordinates [lng, lat] for ORS
        coords = [loc.to_coords() for loc in all_locations]
        
        n_origins = len(origins)
        n_dests = len(destinations)
        
        # Source indices (0 to n_origins-1)
        sources = list(range(n_origins))
        # Destination indices (n_origins to end)
        destinations_idx = list(range(n_origins, n_origins + n_dests))
        
        print(f"  Calling ORS Matrix API for {n_origins}x{n_dests} pairs...")
        
        try:
            result = self.client.distance_matrix(
                locations=coords,
                sources=sources,
                destinations=destinations_idx,
                profile=profile,
                metrics=["distance", "duration"]
            )
            
            # Extract matrices
            distances_m = np.array(result["distances"])  # meters
            durations_s = np.array(result["durations"])  # seconds
            
            # Convert to km and hours
            distance_matrix = distances_m / 1000
            time_matrix = durations_s / 3600
            
            # Cache results
            self._cache[cache_key] = {
                "distances": distance_matrix.tolist(),
                "durations": time_matrix.tolist()
            }
            self._save_cache()
            
            # Rate limiting: ORS free tier has limits
            time.sleep(1)  # Be nice to the API
            
            return distance_matrix, time_matrix
            
        except Exception as e:
            print(f"  ORS API error: {e}")
            print("  Falling back to Haversine estimates...")
            return self._compute_haversine_matrix(origins, destinations)
    
    def _compute_haversine_matrix(
        self,
        origins: List[ORSLocation],
        destinations: List[ORSLocation]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: compute straight-line distances with road factor."""
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Earth radius in km
            return c * r * 1.3  # 1.3x road factor
        
        n_origins = len(origins)
        n_dests = len(destinations)
        
        distance_matrix = np.zeros((n_origins, n_dests))
        time_matrix = np.zeros((n_origins, n_dests))
        
        for i, origin in enumerate(origins):
            for j, dest in enumerate(destinations):
                dist = haversine(origin.lat, origin.lng, dest.lat, dest.lng)
                distance_matrix[i, j] = dist
                time_matrix[i, j] = dist / 50  # 50 km/h average
        
        return distance_matrix, time_matrix


def compute_ors_distance_matrices(
    network,
    supplier_coords: List[Tuple[float, float]],  # (lat, lng) tuples
    warehouse_coords: List[Tuple[float, float]],
    customer_coords: List[Tuple[float, float]],
    hours_per_day: float = 8.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distance and lead time matrices using OpenRouteService.
    
    Args:
        network: SupplyChainNetwork to update
        supplier_coords: (lat, lng) tuples for suppliers
        warehouse_coords: (lat, lng) tuples for warehouses
        customer_coords: (lat, lng) tuples for customers
        hours_per_day: Driving hours per day
        
    Returns:
        Tuple of (D_SW, D_WC, L_SW, L_WC)
    """
    client = OpenRouteServiceClient()
    
    # Convert to ORS locations
    supplier_locs = [
        ORSLocation(network.suppliers[i].name, lng, lat) 
        for i, (lat, lng) in enumerate(supplier_coords)
    ]
    warehouse_locs = [
        ORSLocation(network.warehouses[j].name, lng, lat)
        for j, (lat, lng) in enumerate(warehouse_coords)
    ]
    customer_locs = [
        ORSLocation(network.customers[k].name, lng, lat)
        for k, (lat, lng) in enumerate(customer_coords)
    ]
    
    print("[OpenRouteService] Computing supplier -> warehouse distances...")
    d_sw, t_sw = client.compute_distance_matrix(supplier_locs, warehouse_locs, "driving-hgv")
    
    print("[OpenRouteService] Computing warehouse -> customer distances...")
    d_wc, t_wc = client.compute_distance_matrix(warehouse_locs, customer_locs, "driving-hgv")
    
    # Convert travel time to lead time (days)
    l_sw = np.ceil(t_sw / hours_per_day).astype(int)
    l_sw = np.maximum(l_sw, 1)
    
    l_wc = np.ceil(t_wc / hours_per_day).astype(int)
    l_wc = np.maximum(l_wc, 1)
    
    # Update network
    network.distance_sw = d_sw
    network.distance_wc = d_wc
    network.lead_time_sw = l_sw
    network.lead_time_wc = l_wc
    
    print(f"  D_SW: {d_sw.shape}, avg distance = {d_sw.mean():.1f} km")
    print(f"  D_WC: {d_wc.shape}, avg distance = {d_wc.mean():.1f} km")
    print(f"  L_SW: avg lead time = {l_sw.mean():.1f} days")
    print(f"  L_WC: avg lead time = {l_wc.mean():.1f} days")
    
    return d_sw, d_wc, l_sw, l_wc
