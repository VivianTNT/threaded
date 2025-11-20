import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

async function checkCounts() {
  const { count: projects } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  const { count: companies } = await supabase.from('companies').select('*', { count: 'exact', head: true });
  const { count: news } = await supabase.from('news').select('*', { count: 'exact', head: true });

  console.log(`📊 Database Counts:`);
  console.log(`   Projects: ${projects}`);
  console.log(`   Companies: ${companies}`);
  console.log(`   News: ${news}`);
}

checkCounts();
