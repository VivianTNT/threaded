# FactSet Global Filings API - Mining Projects Population Script

## Overview

This script uses the **FactSet Global Filings API** to automatically populate your Supabase `projects` table with mining project data extracted from **SEDAR (Canadian Securities)** filings.

## What It Does

1. **Loads Canadian Mining Companies** from your local data file (`data/mining-companies-comprehensive.json`)
2. **Converts Tickers to FactSet Format**:
   - TSX: `ABC` → `ABC-TOR`
   - TSXV: `XYZ` → `XYZ-TOV`
   - NYSE/NASDAQ: `DEF` → `DEF-US`
3. **Searches SEDAR+ Filings** from 2025 onwards for each company
4. **Extracts Project Names** from filing headlines (especially NI 43-101 technical reports)
5. **Populates Supabase** with project name and filing URLs

## Prerequisites

### 1. FactSet API Access

You need a **FactSet account with Global Filings API access**. Contact FactSet to obtain:
- Username (your FactSet serial number)
- API Key

### 2. Environment Variables

Add the following to your `.env.local` file:

```bash
# FactSet Credentials
FACTSET_USERNAME=your-factset-username
FACTSET_API_KEY=your-factset-api-key

# Supabase (should already exist)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## Installation

No additional packages needed - uses existing project dependencies:
- `@supabase/supabase-js`
- `dotenv`
- `node-fetch` (built into Node 18+)

## Usage

### Run the Script

```bash
npx tsx scripts/populate-projects-from-factset.ts
```

### Expected Output

```
======================================================================
FACTSET SEDAR FILINGS → SUPABASE PROJECTS POPULATION
======================================================================

✅ FactSet credentials configured
📡 API Base URL: https://api.factset.com/global-filings/v2
🔑 Username: YOUR-USERNAME

📂 Loading Canadian mining companies...
✅ Loaded 20 Canadian mining companies

📊 Loading companies from Supabase...
✅ Loaded 150 companies from database

======================================================================
PROCESSING COMPANIES
======================================================================

[1/10] Ivanhoe Mines Ltd. (IVN → IVN-TOR)
   ✅ Found 12 filing(s)
   📋 Extracted 3 project(s)
      ✅ Inserted: Kamoa-Kakula Copper Project
      ✅ Inserted: Platreef Project
      ✅ Inserted: Kipushi Project

[2/10] First Quantum Minerals Ltd. (FM → FM-TOR)
   ✅ Found 8 filing(s)
   📋 Extracted 2 project(s)
      ✅ Inserted: Cobre Panama Project
      ⏭️  Skipped: Kansanshi Mine (already exists)

...

======================================================================
POPULATION COMPLETE
======================================================================

📊 Summary:
   Companies processed: 10
   Companies with filings: 7
   Total filings found: 45
   Total projects extracted: 18
   Projects inserted: 15
   Projects skipped (duplicates): 3
   Errors: 0

   Total projects in database: 215

✅ Process complete!
```

## How It Works

### 1. Ticker Conversion

Canadian companies listed on TSX/TSXV need to be converted to FactSet's ticker format:

```typescript
TSX/TSXV: IVN  → IVN-CA    (Canadian companies)
NYSE:     LAC  → LAC-US    (US companies)
ASX:      PLS  → PLS-AU    (Australian companies)
LSE:      RIO  → RIO-GB    (UK companies)
```

**Note**: FactSet uses country codes (-CA, -US, -AU, -GB) rather than exchange-specific codes.

### 2. API Search Parameters

The script calls the `/search` endpoint with:

```javascript
{
  ids: ["IVN-TOR"],           // FactSet ticker
  sources: ["SDRP"],          // SEDAR+ (post-Sept 2024)
  startDate: "20250101",      // From 2025 onwards
  _paginationLimit: 50,       // Max 50 results per company
  timeZone: "America/Toronto",
  _sort: ["-filingsDateTime"] // Most recent first
}
```

### 3. Project Extraction

The script looks for mining-specific patterns in filing headlines:

- **NI 43-101 Technical Reports** (Canadian mining standard)
- Mentions of "Project", "Property", "Mine", "Deposit"
- Annual/Quarterly reports mentioning specific projects

Example headline:
```
"NI 43-101 Technical Report for the Kamoa-Kakula Copper Project"
→ Extracts: "Kamoa-Kakula Copper Project"
```

### 4. Database Insertion

For each extracted project:
1. Match company by ticker to get `company_id`
2. Check if project already exists (by name + company_id)
3. Insert with fields:
   - `name`: Extracted project name
   - `urls`: Array containing filing link(s)
   - `company_id`: Foreign key to companies table
   - `description`: Brief description from filing
   - `status`: Set to 'Active'

## Rate Limiting

- **FactSet API Limit**: 10 requests/second
- **Script Delay**: 150ms between requests (safe buffer)
- Processing 100 companies takes ~15 seconds

## Customization

### Process More Companies

By default, the script processes the first 10 companies (for testing). To process all:

```typescript
// In main() function, change:
const companiesToProcess = companies.slice(0, 10)

// To:
const companiesToProcess = companies  // Process all
```

### Adjust Date Range

```typescript
// Search filings from different date range:
const filings = await searchSedarFilings(
  factsetTicker,
  '20240101',  // Start date
  '20241231'   // End date (optional)
)
```

### Add More Sources

To include SEDAR (pre-2024) filings:

```typescript
sources: ['SDR', 'SDRP']  // Both old and new SEDAR
```

### Filter by Form Types

To target specific filing types:

```typescript
const params = {
  // ... existing params
  formTypes: ['NI 43-101']  // Only technical reports
}
```

## Troubleshooting

### Error: "FactSet credentials not configured"

**Solution**: Add `FACTSET_USERNAME` and `FACTSET_API_KEY` to `.env.local`

### Error: "401 Unauthorized"

**Solution**: Check your FactSet credentials are correct and your account has API access

### Error: "Company not found in database"

**Solution**: The ticker exists in your JSON file but not in Supabase `companies` table. Run `populate-mining-companies.ts` first.

### No filings found for most companies

**Possible causes**:
1. Companies may not have filed recently (2025)
2. Ticker format may be incorrect
3. Companies may not file with SEDAR (check if they're truly Canadian)

**Solution**: Try expanding date range or adding more sources (EDGAR for US-listed Canadian companies)

### Projects not being extracted

**Possible causes**:
1. Filings may not mention specific project names
2. Pattern matching may need adjustment for different headline formats

**Solution**: Check the filing headlines in the output and adjust the regex patterns in `extractProjectsFromFiling()`

## API Reference

### FactSet Global Filings API v2

**Base URL**: `https://api.factset.com/global-filings/v2`

**Authentication**: Basic Auth (username:api_key)

**Key Endpoints**:
- `GET /search` - Search for filings
- `GET /count` - Get filing counts
- `GET /meta/sources` - List available sources

**Documentation**: https://developer.factset.com/api-catalog/global-filings-api

### SEDAR Sources

- **SDR**: SEDAR (pre-September 30, 2024)
- **SDRP**: SEDAR+ (post-September 30, 2024)

## Mining-Specific Form Types

Common Canadian mining filing types to look for:

- **NI 43-101**: Technical Reports (most valuable for project data)
- **AIF**: Annual Information Form
- **MD&A**: Management Discussion & Analysis
- **Press Releases**: Project updates and announcements

## Next Steps

After running this script:

1. **Review inserted projects** in your Supabase dashboard
2. **Manually verify** project names are accurate
3. **Enhance projects** by running other scrapers to fill in additional fields:
   - Location, stage, commodities
   - Financial metrics (NPV, IRR, CAPEX)
   - Resource/reserve estimates

## Future Enhancements

Potential improvements:

1. **Download PDF documents** from filing links and extract more detailed data
2. **NLP/GPT extraction** to parse technical reports for financial metrics
3. **Multi-exchange support** (expand to ASX, LSE using respective sources)
4. **Form type filtering** to prioritize NI 43-101 technical reports
5. **Incremental updates** to check for new filings periodically

## Related Scripts

- `populate-mining-companies.ts` - Populate companies table first
- `populate-mining-projects.ts` - Alternative method using static data
- `extract-mining-companies.ts` - Extract companies using Firecrawl

## Support

For FactSet API issues:
- **Developer Portal**: https://developer.factset.com
- **Support**: Contact your FactSet account representative

For script issues:
- Check the error messages in console output
- Verify your data files exist in the `data/` directory
- Ensure Supabase is properly configured
