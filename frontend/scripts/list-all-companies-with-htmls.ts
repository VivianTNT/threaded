import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);

async function listCompaniesWithHtmls() {
  console.log('📂 Listing all companies with HTML documents...\n');

  // Recursively search for HTML files
  const companyMap = new Map<string, string[]>();

  async function searchFolder(prefix: string) {
    const { data, error } = await supabase.storage
      .from('refinitiv')
      .list(prefix, { limit: 1000 });

    if (error) return;

    for (const item of data || []) {
      const path = prefix ? `${prefix}/${item.name}` : item.name;

      if (item.name.endsWith('.html')) {
        // Extract company name from path (e.g., "factset/edg/barrick-gold/2025/file.html")
        const parts = prefix.split('/');
        if (parts.length >= 3) {
          const company = parts[2];
          if (!companyMap.has(company)) {
            companyMap.set(company, []);
          }
          companyMap.get(company)!.push(path);
        }
      } else if (!item.name.includes('.')) {
        await searchFolder(path);
      }
    }
  }

  await searchFolder('factset');

  console.log(`Found ${companyMap.size} companies with HTML documents:\n`);

  // Sort by number of documents
  const sorted = Array.from(companyMap.entries()).sort((a, b) => b[1].length - a[1].length);

  for (const [company, docs] of sorted) {
    console.log(`📊 ${company}: ${docs.length} documents`);
    docs.slice(0, 3).forEach(doc => {
      const filename = doc.split('/').pop();
      const url = `https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/${doc}`;
      console.log(`   - ${filename}`);
      console.log(`     ${url}`);
    });
    if (docs.length > 3) {
      console.log(`   ... and ${docs.length - 3} more`);
    }
    console.log('');
  }
}

listCompaniesWithHtmls().catch(console.error);
