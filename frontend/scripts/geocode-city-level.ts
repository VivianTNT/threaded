import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Sleep function to respect rate limits
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Geocode using Nominatim (OpenStreetMap) - free, no API key needed
async function geocodeLocation(location: string): Promise<{ lat: number; lng: number } | null> {
  try {
    const encodedLocation = encodeURIComponent(location);
    const url = `https://nominatim.openstreetmap.org/search?q=${encodedLocation}&format=json&limit=1`;

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'LithosMiningApp/1.0' // Required by Nominatim
      }
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();

    if (data && data.length > 0) {
      return {
        lat: parseFloat(data[0].lat),
        lng: parseFloat(data[0].lon)
      };
    }

    return null;
  } catch (error) {
    console.error(`Error geocoding ${location}:`, error);
    return null;
  }
}

async function geocodeAllProjectsCityLevel() {
  console.log('🌍 Starting city-level geocoding of all projects...\n');

  let totalUpdated = 0;
  let totalFailed = 0;
  let totalSkipped = 0;
  let offset = 0;
  const pageSize = 100; // Smaller batches for rate limiting

  while (true) {
    // Fetch batch of projects without coordinates
    const { data: projects, error } = await supabase
      .from('projects')
      .select('id, name, location, latitude, longitude')
      .is('latitude', null)
      .not('location', 'is', null)
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

    for (let i = 0; i < projects.length; i++) {
      const project = projects[i];

      if (!project.location) {
        totalSkipped++;
        continue;
      }

      // Geocode the location
      const coords = await geocodeLocation(project.location);

      if (coords) {
        // Update project with coordinates
        const { error: updateError } = await supabase
          .from('projects')
          .update({
            latitude: coords.lat,
            longitude: coords.lng
          })
          .eq('id', project.id);

        if (updateError) {
          console.error(`  ❌ Failed to update ${project.name}`);
          totalFailed++;
        } else {
          totalUpdated++;
        }
      } else {
        totalFailed++;
      }

      // Progress every 10 projects
      if ((i + 1) % 10 === 0) {
        console.log(`  Progress: ${i + 1}/${projects.length} (${totalUpdated} updated, ${totalFailed} failed, ${totalSkipped} skipped)`);
      }

      // Rate limiting: 1 request per second (Nominatim requirement)
      await sleep(1000);
    }

    console.log(`Batch complete: ${totalUpdated} total updated, ${totalFailed} total failed, ${totalSkipped} total skipped`);

    // If we got less than pageSize, we're done
    if (projects.length < pageSize) {
      break;
    }

    offset += pageSize;
  }

  console.log(`\n✅ City-level geocoding complete!`);
  console.log(`   Total updated: ${totalUpdated}`);
  console.log(`   Total failed: ${totalFailed}`);
  console.log(`   Total skipped: ${totalSkipped}`);
}

geocodeAllProjectsCityLevel();
