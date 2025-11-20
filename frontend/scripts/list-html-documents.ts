import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);

async function listHtmlDocuments() {
  console.log('📂 Listing HTML documents in Supabase storage...\n');

  // Recursively search for HTML files
  const htmlFiles: string[] = [];

  async function searchFolder(prefix: string) {
    const { data, error } = await supabase.storage
      .from('refinitiv')
      .list(prefix, { limit: 1000 });

    if (error) {
      console.error(`Error listing ${prefix}:`, error);
      return;
    }

    for (const item of data || []) {
      const path = prefix ? `${prefix}/${item.name}` : item.name;

      if (item.name.endsWith('.html')) {
        htmlFiles.push(path);
      } else if (!item.name.includes('.')) {
        // It's a folder, search recursively
        await searchFolder(path);
      }
    }
  }

  await searchFolder('factset');

  console.log(`Found ${htmlFiles.length} HTML documents:\n`);

  // Group by type
  const sedarDocs = htmlFiles.filter(f => f.toLowerCase().includes('sedar') || f.includes('canada'));
  const edgarDocs = htmlFiles.filter(f => f.includes('edg') && !sedarDocs.includes(f));

  console.log(`📊 SEDAR Documents (${sedarDocs.length}):\n`);
  sedarDocs.slice(0, 20).forEach((file, i) => {
    const url = `https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/${file}`;
    const filename = file.split('/').pop();
    const company = file.split('/')[2] || 'unknown';
    console.log(`${i + 1}. ${filename}`);
    console.log(`   Company folder: ${company}`);
    console.log(`   Full URL: ${url}\n`);
  });

  console.log(`\n📊 EDGAR Documents (${edgarDocs.length}):\n`);
  edgarDocs.slice(0, 10).forEach((file, i) => {
    const filename = file.split('/').pop();
    const company = file.split('/')[2] || 'unknown';
    const url = `https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/${file}`;
    console.log(`${i + 1}. ${filename}`);
    console.log(`   Company folder: ${company}`);
    console.log(`   Full URL: ${url}\n`);
  });
}

listHtmlDocuments().catch(console.error);
