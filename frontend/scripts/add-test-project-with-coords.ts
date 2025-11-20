import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function addTestProject() {
  console.log('Creating test project with coordinates...\n');

  const testProject = {
    name: 'Red Lake Gold Mine TEST',
    location: 'Red Lake, Ontario, Canada',
    latitude: 51.0456,
    longitude: -93.8294,
    stage: 'Production',
    status: 'Active',
    commodities: ['Gold'],
    description: 'High-grade underground gold mine in the Red Lake mining district',

    // Financial Metrics (values already in millions for NPV/CAPEX)
    npv: 892.5, // $892.5M
    irr: 28.5,
    capex: 445, // $445M
    aisc: 925, // $925/oz

    // Resource/Reserve
    resource: '5.2M oz Au @ 10.5 g/t',
    reserve: '2.8M oz Au @ 9.8 g/t',

    // Qualified Persons
    qualified_persons: [
      {
        name: 'Dr. Jennifer Mills',
        credentials: 'P.Eng., Ph.D. (Mining Engineering)',
        company: 'Evolution Mining'
      },
      {
        name: 'Robert Thompson',
        credentials: 'P.Geo., M.Sc.',
        company: 'SRK Consulting'
      }
    ]
  };

  const { data, error } = await supabase
    .from('projects')
    .insert([testProject])
    .select();

  if (error) {
    console.error('Error creating test project:', error);
    return;
  }

  console.log('✅ Test project created successfully!\n');
  console.log('Project Details:');
  console.log(`  Name: ${testProject.name}`);
  console.log(`  Location: ${testProject.location}`);
  console.log(`  Coordinates: ${testProject.latitude}, ${testProject.longitude}`);
  console.log(`  NPV: $${testProject.npv}M`);
  console.log(`  IRR: ${testProject.irr}%`);
  console.log(`  AISC: $${testProject.aisc}/oz`);
  console.log(`  CAPEX: $${testProject.capex}M`);
  console.log(`  Qualified Persons: ${testProject.qualified_persons?.length || 0}`);
  console.log('\nThis project will show:');
  console.log('  ✓ AISC in Sensitivity Analysis: $925.50');
  console.log('  ✓ Latitude in Project Info: 51.0456');
  console.log('  ✓ Longitude in Project Info: -93.8294');
  console.log('  ✓ 2D Map pinpointing Red Lake, Ontario at exact coordinates');
  console.log('  ✓ 3D Globe pin at exact coordinates (51.0456°N, 93.8294°W)');
  console.log('\n🔍 Search for "Red Lake" in the projects table to view it!');
}

addTestProject();
