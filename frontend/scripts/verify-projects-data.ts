import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  throw new Error('Missing Supabase environment variables');
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false
  }
});

async function verifyProjectsData() {
  console.log('🔍 Verifying projects data quality...\n');

  try {
    // Get total count and stats
    const { data: allProjects, count } = await supabase
      .from('projects')
      .select('*', { count: 'exact' });

    console.log(`📊 TOTAL PROJECTS: ${count}\n`);

    if (!allProjects) {
      console.error('No projects found!');
      return;
    }

    // Analyze commodities
    const commodityCounts: { [key: string]: number } = {};
    allProjects.forEach(p => {
      if (p.commodities && Array.isArray(p.commodities)) {
        p.commodities.forEach((c: string) => {
          commodityCounts[c] = (commodityCounts[c] || 0) + 1;
        });
      }
    });

    console.log('🌍 TOP COMMODITIES:');
    Object.entries(commodityCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .forEach(([commodity, count]) => {
        console.log(`   ${commodity.padEnd(20)} ${count} projects`);
      });

    // Analyze stages
    const stageCounts: { [key: string]: number } = {};
    allProjects.forEach(p => {
      const stage = p.stage || 'Unknown';
      stageCounts[stage] = (stageCounts[stage] || 0) + 1;
    });

    console.log('\n📈 PROJECTS BY STAGE:');
    Object.entries(stageCounts)
      .sort((a, b) => b[1] - a[1])
      .forEach(([stage, count]) => {
        console.log(`   ${stage.padEnd(25)} ${count} projects (${((count / (count || 1)) * 100).toFixed(1)}%)`);
      });

    // Analyze locations
    const locationCounts: { [key: string]: number } = {};
    allProjects.forEach(p => {
      const location = p.location || 'Unknown';
      locationCounts[location] = (locationCounts[location] || 0) + 1;
    });

    console.log('\n🗺️  TOP LOCATIONS:');
    Object.entries(locationCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .forEach(([location, count]) => {
        console.log(`   ${location.padEnd(30)} ${count} projects`);
      });

    // Analyze status
    const statusCounts: { [key: string]: number } = {};
    allProjects.forEach(p => {
      const status = p.status || 'Unknown';
      statusCounts[status] = (statusCounts[status] || 0) + 1;
    });

    console.log('\n✅ PROJECTS BY STATUS:');
    Object.entries(statusCounts)
      .sort((a, b) => b[1] - a[1])
      .forEach(([status, count]) => {
        console.log(`   ${status.padEnd(20)} ${count} projects`);
      });

    // Data quality checks
    const withResources = allProjects.filter(p => p.resource_estimate).length;
    const withReserves = allProjects.filter(p => p.reserve_estimate).length;
    const withDescription = allProjects.filter(p => p.description).length;
    const withCompanyId = allProjects.filter(p => p.company_id).length;

    console.log('\n📋 DATA QUALITY METRICS:');
    console.log(`   Projects with Resource Estimate:  ${withResources} (${((withResources / count!) * 100).toFixed(1)}%)`);
    console.log(`   Projects with Reserve Estimate:   ${withReserves} (${((withReserves / count!) * 100).toFixed(1)}%)`);
    console.log(`   Projects with Description:        ${withDescription} (${((withDescription / count!) * 100).toFixed(1)}%)`);
    console.log(`   Projects with Company Link:       ${withCompanyId} (${((withCompanyId / count!) * 100).toFixed(1)}%)`);

    // Sample projects
    console.log('\n📰 SAMPLE PROJECTS (Random 10):');
    console.log('═══════════════════════════════════════════════════════════════');

    const sampleProjects = allProjects
      .sort(() => Math.random() - 0.5)
      .slice(0, 10);

    sampleProjects.forEach((p, i) => {
      console.log(`\n${i + 1}. ${p.name}`);
      console.log(`   Commodities: ${p.commodities?.join(', ') || 'N/A'}`);
      console.log(`   Location: ${p.location || 'N/A'}`);
      console.log(`   Stage: ${p.stage || 'N/A'}`);
      console.log(`   Status: ${p.status || 'N/A'}`);
      console.log(`   Ownership: ${p.ownership_percentage ? p.ownership_percentage + '%' : 'N/A'}`);
      if (p.resource_estimate) {
        console.log(`   Resources: ${p.resource_estimate}`);
      }
      if (p.reserve_estimate) {
        console.log(`   Reserves: ${p.reserve_estimate}`);
      }
      console.log(`   Description: ${p.description?.substring(0, 100)}...`);
    });

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('✅ Data quality verification complete!\n');

  } catch (error) {
    console.error('❌ Error:', error);
    throw error;
  }
}

verifyProjectsData()
  .then(() => {
    console.log('✅ Verification complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Failed:', error);
    process.exit(1);
  });
