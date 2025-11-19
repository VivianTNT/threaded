#!/usr/bin/env tsx

import { createClient } from '@supabase/supabase-js';
import { config } from 'dotenv';
import path from 'path';

config({ path: path.join(__dirname, '..', '.env.local') });

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function main() {
  console.log('Deleting bad FactSet news entries...');

  const { data, error } = await supabase
    .from('news')
    .delete()
    .eq('title', 'No title')
    .eq('source', 'FactSet StreetAccount')
    .select();

  if (error) {
    console.error('Error:', error);
  } else {
    console.log(`✓ Deleted ${data?.length || 0} bad entries`);
  }
}

main();
