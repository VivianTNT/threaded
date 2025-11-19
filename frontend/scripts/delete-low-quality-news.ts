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
  console.log('Deleting low-quality news headlines...\n');

  const lowQualityPatterns = [
    'StreetAccount Top Stories',
    'StreetAccount Summary',
    'StreetAccount US Evening Reads',
    'StreetAccount Crypto Summary',
    'StreetAccount Scorecard',
    'Market Synopsis',
    'market recap'
  ];

  let totalDeleted = 0;

  for (const pattern of lowQualityPatterns) {
    console.log(`Deleting headlines matching: "${pattern}"`);

    const { data, error } = await supabase
      .from('news')
      .delete()
      .ilike('title', `%${pattern}%`)
      .select();

    if (error) {
      console.error(`  ✗ Error: ${error.message}`);
    } else {
      const count = data?.length || 0;
      totalDeleted += count;
      console.log(`  ✓ Deleted ${count} headlines`);
    }
  }

  console.log(`\n✓ Total deleted: ${totalDeleted} low-quality headlines`);
}

main();
