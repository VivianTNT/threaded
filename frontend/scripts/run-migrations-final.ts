import { Client } from 'pg';
import * as fs from 'fs';
import * as path from 'path';

// Extract project ref from Supabase URL
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const PROJECT_REF = SUPABASE_URL.replace('https://', '').split('.')[0];
const DB_PASSWORD = process.env.DATABASE_PASSWORD || process.env.SUPABASE_DB_PASSWORD;

if (!DB_PASSWORD) {
  console.error('\n❌ DATABASE_PASSWORD not found in environment variables!');
  console.error('\nTo get your database password:');
  console.error('1. Go to: https://supabase.com/dashboard/project/' + PROJECT_REF + '/settings/database');
  console.error('2. Click "Reset database password"');
  console.error('3. Copy the password and add it to your .env.local file:');
  console.error('   DATABASE_PASSWORD=your_password_here\n');
  process.exit(1);
}

// Try multiple connection methods
const connectionConfigs = [
  {
    name: 'Session Pooler (Port 5432)',
    connectionString: `postgresql://postgres.${PROJECT_REF}:${DB_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:5432/postgres`,
  },
  {
    name: 'Transaction Pooler (Port 6543)',
    connectionString: `postgresql://postgres.${PROJECT_REF}:${DB_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:6543/postgres?pgbouncer=true`,
  },
  {
    name: 'Direct Connection',
    connectionString: `postgresql://postgres:${DB_PASSWORD}@db.${PROJECT_REF}.supabase.co:5432/postgres`,
  },
];

async function testConnection(config: typeof connectionConfigs[0]): Promise<Client | null> {
  const client = new Client({
    connectionString: config.connectionString,
    ssl: { rejectUnauthorized: false },
    connectionTimeoutMillis: 5000,
  });

  try {
    console.log(`\nTrying: ${config.name}...`);
    await client.connect();
    console.log(`✅ Connected successfully via ${config.name}!`);
    return client;
  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}`);
    return null;
  }
}

async function runMigrations() {
  let client: Client | null = null;

  // Try each connection method until one works
  for (const config of connectionConfigs) {
    client = await testConnection(config);
    if (client) break;
  }

  if (!client) {
    console.error('\n❌ Could not connect to database with any method.');
    console.error('Please verify:');
    console.error('1. Your DATABASE_PASSWORD is correct');
    console.error('2. Your Supabase project region (check dashboard)');
    console.error('3. Your network allows connections to Supabase');
    process.exit(1);
  }

  try {
    // Get only the new migration files we want to run
    const migrationFiles = [
      '006_create_companies_table.sql',
      '007_create_projects_table.sql',
      '008_create_news_table.sql',
    ];

    console.log(`\n📦 Running ${migrationFiles.length} migrations...\n`);

    for (const file of migrationFiles) {
      const filePath = path.join(__dirname, '../supabase/migrations', file);

      if (!fs.existsSync(filePath)) {
        console.log(`⚠️  Skipping ${file} - file not found`);
        continue;
      }

      console.log(`🚀 Running: ${file}`);
      const sql = fs.readFileSync(filePath, 'utf8');

      try {
        await client.query(sql);
        console.log(`✅ ${file} completed successfully\n`);
      } catch (error: any) {
        console.error(`❌ ${file} failed:`);
        console.error(`   ${error.message}\n`);

        // Check if it's just a "already exists" error
        if (error.message.includes('already exists')) {
          console.log(`ℹ️  Table/function already exists, continuing...\n`);
        } else {
          throw error;
        }
      }
    }

    console.log('✨ All migrations completed successfully!\n');

  } catch (error: any) {
    console.error('\n❌ Migration failed:', error.message);
    process.exit(1);
  } finally {
    await client.end();
  }
}

// Run migrations
runMigrations().catch(console.error);
