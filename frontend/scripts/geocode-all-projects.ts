import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Country-level coordinates as fallback
const countryCoords: Record<string, { lat: number; lng: number }> = {
  'USA': { lat: 37.0902, lng: -95.7129 },
  'United States': { lat: 37.0902, lng: -95.7129 },
  'Canada': { lat: 56.1304, lng: -106.3468 },
  'Mexico': { lat: 23.6345, lng: -102.5528 },
  'Chile': { lat: -35.6751, lng: -71.5430 },
  'Peru': { lat: -9.1900, lng: -75.0152 },
  'Brazil': { lat: -14.2350, lng: -51.9253 },
  'Argentina': { lat: -38.4161, lng: -63.6167 },
  'Bolivia': { lat: -16.2902, lng: -63.5887 },
  'Sweden': { lat: 60.1282, lng: 18.6435 },
  'Finland': { lat: 61.9241, lng: 25.7482 },
  'Norway': { lat: 60.4720, lng: 8.4689 },
  'UK': { lat: 55.3781, lng: -3.4360 },
  'United Kingdom': { lat: 55.3781, lng: -3.4360 },
  'Spain': { lat: 40.4637, lng: -3.7492 },
  'Portugal': { lat: 39.3999, lng: -8.2245 },
  'France': { lat: 46.2276, lng: 2.2137 },
  'Germany': { lat: 51.1657, lng: 10.4515 },
  'Poland': { lat: 51.9194, lng: 19.1451 },
  'South Africa': { lat: -30.5595, lng: 22.9375 },
  'Mali': { lat: 17.5707, lng: -3.9962 },
  'DRC': { lat: -4.0383, lng: 21.7587 },
  'Congo': { lat: -4.0383, lng: 21.7587 },
  'Zambia': { lat: -13.1339, lng: 27.8493 },
  'Tanzania': { lat: -6.3690, lng: 34.8888 },
  'Namibia': { lat: -22.9576, lng: 18.4904 },
  'Botswana': { lat: -22.3285, lng: 24.6849 },
  'Zimbabwe': { lat: -19.0154, lng: 29.1549 },
  'China': { lat: 35.8617, lng: 104.1954 },
  'Mongolia': { lat: 46.8625, lng: 103.8467 },
  'Kazakhstan': { lat: 48.0196, lng: 66.9237 },
  'India': { lat: 20.5937, lng: 78.9629 },
  'Indonesia': { lat: -0.7893, lng: 113.9213 },
  'Philippines': { lat: 12.8797, lng: 121.7740 },
  'Australia': { lat: -25.2744, lng: 133.7751 },
  'Papua New Guinea': { lat: -6.3150, lng: 143.9555 },
  'New Zealand': { lat: -40.9006, lng: 174.8860 },
};

// Extract country from malformed location string (takes last part after comma)
function extractCountry(location: string | null): string | null {
  if (!location) return null;
  const parts = location.split(',').map(p => p.trim());
  return parts[parts.length - 1]; // Return last part (should be country)
}

// Get coordinates for a country
function getCountryCoords(country: string): { lat: number; lng: number } | null {
  if (countryCoords[country]) {
    return countryCoords[country];
  }

  // Try partial match
  for (const [key, coords] of Object.entries(countryCoords)) {
    if (country.toLowerCase().includes(key.toLowerCase()) ||
        key.toLowerCase().includes(country.toLowerCase())) {
      return coords;
    }
  }

  return null;
}

async function geocodeAllProjects() {
  console.log('🌍 Starting geocoding of all projects...\n');

  let totalUpdated = 0;
  let totalFailed = 0;
  let offset = 0;
  const pageSize = 1000;

  while (true) {
    // Fetch batch of 1000 projects
    const { data: projects, error } = await supabase
      .from('projects')
      .select('id, name, location')
      .is('latitude', null)
      .range(offset, offset + pageSize - 1);

    if (error) {
      console.error('Error fetching projects:', error);
      break;
    }

    if (!projects || projects.length === 0) {
      console.log('No more projects to geocode\n');
      break;
    }

    console.log(`\nProcessing batch: offset ${offset}, count ${projects.length}`);

    let batchUpdated = 0;
    let batchFailed = 0;

    for (const project of projects) {
      const country = extractCountry(project.location);
      if (!country) {
        batchFailed++;
        continue;
      }

      const coords = getCountryCoords(country);
      if (!coords) {
        batchFailed++;
        continue;
      }

      // Update project with country-level coordinates
      const { error: updateError } = await supabase
        .from('projects')
        .update({
          latitude: coords.lat,
          longitude: coords.lng
        })
        .eq('id', project.id);

      if (updateError) {
        batchFailed++;
      } else {
        batchUpdated++;
      }

      // Progress every 100 projects
      if ((batchUpdated + batchFailed) % 100 === 0) {
        console.log(`  Progress: ${batchUpdated + batchFailed}/${projects.length} (${batchUpdated} updated, ${batchFailed} failed)`);
      }
    }

    totalUpdated += batchUpdated;
    totalFailed += batchFailed;

    console.log(`Batch complete: ${batchUpdated} updated, ${batchFailed} failed`);
    console.log(`Total so far: ${totalUpdated} updated, ${totalFailed} failed`);

    // If we got less than pageSize, we're done
    if (projects.length < pageSize) {
      break;
    }

    offset += pageSize;
  }

  console.log(`\n✅ Geocoding complete!`);
  console.log(`   Total updated: ${totalUpdated}`);
  console.log(`   Total failed: ${totalFailed}`);
}

geocodeAllProjects();
