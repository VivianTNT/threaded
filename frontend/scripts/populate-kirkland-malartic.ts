import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function populateKirklandMalartic() {
  console.log('Starting to populate Kirkland Lake Gold and Canadian Malartic companies...\n');

  try {
    // First, check existing companies
    const { data: existingCompanies, error: checkError } = await supabase
      .from('companies')
      .select('id, name, ticker')
      .or('name.ilike.%kirkland%,name.ilike.%malartic%');

    if (checkError) {
      console.error('Error checking existing companies:', checkError);
    } else {
      console.log('Existing companies:', existingCompanies);
    }

    // Kirkland Lake Gold data
    // Note: Kirkland Lake Gold was acquired by Agnico Eagle in 2022
    const kirklandData = {
      name: 'Kirkland Lake Gold Ltd.',
      ticker: 'KL.TO', // Former TSX ticker (now delisted after acquisition)
      exchange: 'TSX',
      market_cap: null, // No longer trading (acquired by Agnico Eagle in 2022)
      country: 'Canada',
      website: 'https://www.agnicoeagle.com',
      description: 'Kirkland Lake Gold was a Canadian gold mining company that operated mines in Canada and Australia. It was acquired by Agnico Eagle Mines in 2022 for approximately $10.6 billion.',
      urls: ['https://www.agnicoeagle.com', 'https://www.mining.com/agnico-eagle-completes-kirkland-lake-gold-acquisition/']
    };

    // Canadian Malartic GP (Partnership between Agnico Eagle and Yamana Gold)
    const malarticData = {
      name: 'Canadian Malartic GP',
      ticker: null, // Joint venture, not publicly traded separately
      exchange: 'Joint Venture',
      market_cap: null,
      country: 'Canada',
      website: 'https://www.agnicoeagle.com/operations/operations-northern-canada/canadian-malartic-mine',
      description: 'Canadian Malartic is one of Canada\'s largest open-pit gold mines, operated as a 50/50 partnership between Agnico Eagle Mines and Yamana Gold. The mine is located in Malartic, Quebec.',
      urls: ['https://www.agnicoeagle.com/operations/operations-northern-canada/canadian-malartic-mine', 'https://canadianmalartic.com/en/']
    };

    // Update Kirkland Lake Gold
    const { data: existingKirkland } = await supabase
      .from('companies')
      .select('id')
      .eq('name', 'Kirkland Lake Gold Ltd.')
      .single();

    let kirklandCompany;
    if (existingKirkland) {
      const { data, error: kirklandError } = await supabase
        .from('companies')
        .update(kirklandData)
        .eq('id', existingKirkland.id)
        .select()
        .single();

      if (kirklandError) {
        console.error('Error updating Kirkland Lake Gold:', kirklandError);
      } else {
        kirklandCompany = data;
        console.log('\n✓ Kirkland Lake Gold company updated:', kirklandCompany);
      }
    } else {
      const { data, error: kirklandError } = await supabase
        .from('companies')
        .insert(kirklandData)
        .select()
        .single();

      if (kirklandError) {
        console.error('Error inserting Kirkland Lake Gold:', kirklandError);
      } else {
        kirklandCompany = data;
        console.log('\n✓ Kirkland Lake Gold company created:', kirklandCompany);
      }
    }

    // Update Canadian Malartic GP
    const { data: existingMalartic } = await supabase
      .from('companies')
      .select('id')
      .eq('name', 'Canadian Malartic GP')
      .single();

    let malarticCompany;
    if (existingMalartic) {
      const { data, error: malarticError } = await supabase
        .from('companies')
        .update(malarticData)
        .eq('id', existingMalartic.id)
        .select()
        .single();

      if (malarticError) {
        console.error('Error updating Canadian Malartic GP:', malarticError);
      } else {
        malarticCompany = data;
        console.log('✓ Canadian Malartic GP company updated:', malarticCompany);
      }
    } else {
      const { data, error: malarticError } = await supabase
        .from('companies')
        .insert(malarticData)
        .select()
        .single();

      if (malarticError) {
        console.error('Error inserting Canadian Malartic GP:', malarticError);
      } else {
        malarticCompany = data;
        console.log('✓ Canadian Malartic GP company created:', malarticCompany);
      }
    }

    // Now update projects to reference these companies
    console.log('\n--- Updating project references ---\n');

    // Get projects that might be related
    const { data: projects, error: projectsError } = await supabase
      .from('projects')
      .select('id, name, company_id, location')
      .or('name.ilike.%kirkland%,name.ilike.%malartic%');

    if (projectsError) {
      console.error('Error fetching projects:', projectsError);
      return;
    }

    console.log(`Found ${projects?.length || 0} projects to update\n`);

    // Update each project with the correct company_id
    for (const project of projects || []) {
      let companyId = null;

      if (project.name.toLowerCase().includes('kirkland')) {
        companyId = kirklandCompany?.id;
        console.log(`Linking project "${project.name}" to Kirkland Lake Gold`);
      } else if (project.name.toLowerCase().includes('malartic')) {
        companyId = malarticCompany?.id;
        console.log(`Linking project "${project.name}" to Canadian Malartic GP`);
      }

      if (companyId) {
        const { error: updateError } = await supabase
          .from('projects')
          .update({ company_id: companyId })
          .eq('id', project.id);

        if (updateError) {
          console.error(`  ✗ Error updating project ${project.name}:`, updateError);
        } else {
          console.log(`  ✓ Updated project ${project.name}`);
        }
      }
    }

    // Final verification
    console.log('\n--- Final Verification ---\n');

    const { data: verifyProjects } = await supabase
      .from('projects')
      .select(`
        id,
        name,
        location,
        companies!inner (
          id,
          name,
          ticker,
          exchange
        )
      `)
      .or('name.ilike.%kirkland%,name.ilike.%malartic%');

    console.log('Projects with company references:');
    verifyProjects?.forEach(p => {
      console.log(`  - ${p.name}`);
      console.log(`    Company: ${(p.companies as any).name} (${(p.companies as any).ticker})`);
      console.log(`    Exchange: ${(p.companies as any).exchange}`);
      console.log(`    Location: ${p.location}\n`);
    });

    console.log('\n✓ Population complete!');

  } catch (error) {
    console.error('Unexpected error:', error);
  }
}

populateKirklandMalartic();
