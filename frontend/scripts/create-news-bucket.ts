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
  console.log('Creating news bucket in Supabase storage...\n');

  // Create news bucket
  const { data, error } = await supabase.storage.createBucket('news', {
    public: true,
    fileSizeLimit: 52428800, // 50MB
    allowedMimeTypes: ['application/pdf', 'text/html']
  });

  if (error) {
    if (error.message.includes('already exists')) {
      console.log('✓ News bucket already exists');
    } else {
      console.error('✗ Error creating bucket:', error);
      process.exit(1);
    }
  } else {
    console.log('✓ News bucket created successfully');
  }

  console.log('\nBucket configuration:');
  console.log('- Name: news');
  console.log('- Public: true');
  console.log('- Max file size: 50MB');
  console.log('- Allowed types: PDF, HTML');
}

main();
