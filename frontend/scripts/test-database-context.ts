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

async function testDatabaseContext() {
  console.log('🔍 Testing database context data...\n');

  try {
    // Test projects
    const { data: projects, count: projectCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact' })
      .limit(3);

    console.log(`📊 PROJECTS: ${projectCount} total`);
    if (projects && projects.length > 0) {
      console.log('Sample project:');
      console.log(`  - Name: ${projects[0].project_name}`);
      console.log(`  - Company: ${projects[0].company_name}`);
      console.log(`  - Country: ${projects[0].country}`);
      console.log(`  - Commodity: ${projects[0].primary_commodity}`);
      console.log(`  - NPV: $${projects[0].post_tax_npv_usd_m}M`);
      console.log('');
    }

    // Test companies
    const { data: companies, count: companyCount } = await supabase
      .from('companies')
      .select('*', { count: 'exact' })
      .limit(3);

    console.log(`🏢 COMPANIES: ${companyCount} total`);
    if (companies && companies.length > 0) {
      console.log('Sample company:');
      console.log(`  - Name: ${companies[0].name}`);
      console.log(`  - Ticker: ${companies[0].ticker}`);
      console.log(`  - Exchange: ${companies[0].exchange}`);
      console.log(`  - Country: ${companies[0].country}`);
      console.log(`  - Market Cap: $${companies[0].market_cap ? (companies[0].market_cap / 1000000).toFixed(1) + 'M' : 'N/A'}`);
      console.log('');
    }

    // Test news
    const { data: news, count: newsCount } = await supabase
      .from('news')
      .select('*', { count: 'exact' })
      .limit(3);

    console.log(`📰 NEWS: ${newsCount} total`);
    if (news && news.length > 0) {
      console.log('Sample news:');
      console.log(`  - Title: ${news[0].title}`);
      console.log(`  - Source: ${news[0].source}`);
      console.log(`  - Date: ${news[0].published_at}`);
      console.log(`  - Commodities: ${news[0].commodities?.join(', ') || 'N/A'}`);
      console.log('');
    }

    console.log('✅ Database context data is ready!');
    console.log(`\n📊 Summary: ${projectCount} projects, ${companyCount} companies, ${newsCount} news articles`);

  } catch (error) {
    console.error('❌ Error:', error);
    throw error;
  }
}

testDatabaseContext()
  .then(() => {
    console.log('\n✅ Test complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Failed:', error);
    process.exit(1);
  });
