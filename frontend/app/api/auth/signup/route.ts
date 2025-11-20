import { NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { config } from '@/lib/config'

export async function POST(request: Request) {
  try {
    const { email, password, userData } = await request.json()
    
    // Validate website URL
    if (userData.website) {
      try {
        new URL(userData.website)
      } catch (e) {
        return NextResponse.json(
          { success: false, message: 'Please enter a valid website URL' },
          { status: 400 }
        )
      }
    }
    
    // First, sign up with Supabase Auth
    const { data: authData, error: signUpError } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: config.auth.emailRedirectTo,
      }
    })

    if (signUpError) {
      return NextResponse.json(
        { success: false, message: signUpError.message },
        { status: 400 }
      )
    }

    if (!authData.user) {
      return NextResponse.json(
        { success: false, message: 'Failed to create auth user' },
        { status: 400 }
      )
    }

    // Wait a moment for the auth user to be fully created
    await new Promise(resolve => setTimeout(resolve, 1000))

    // Check if email already exists in usr table
    const { data: existingUsers, error: queryError } = await supabase
      .from('usr')
      .select('email')
      .eq('email', email)
      .limit(1)
    
    if (queryError && queryError.code !== 'PGRST116') { // PGRST116 means no rows found
      console.error('Error checking existing users:', queryError)
    }
    
    if (existingUsers && existingUsers.length > 0) {
      return NextResponse.json(
        { success: false, message: 'Email already taken' },
        { status: 400 }
      )
    }

    // Check if brand exists or create new one
    let brandId: number
    const brandCode = userData.brand.toLowerCase().replace(/[^a-z0-9]+/g, '-')
    
    // First try to find existing brand by code
    const { data: existingBrand, error: brandQueryError } = await supabase
      .from('brands')
      .select('brand_id')
      .eq('brand_code', brandCode)
      .maybeSingle()
    
    if (brandQueryError && brandQueryError.code !== 'PGRST116') {
      console.error('Error checking existing brand:', brandQueryError)
      return NextResponse.json(
        { success: false, message: 'Failed to check brand' },
        { status: 500 }
      )
    }
    
    if (existingBrand) {
      brandId = existingBrand.brand_id
    } else {
      // Create new brand
      const { data: newBrand, error: brandError } = await supabase
        .from('brands')
        .insert({
          brand_name: userData.brand,
          brand_code: brandCode,
          description: userData.desc,
          category: userData.category,
          website: userData.website,
          founded_year: userData.founded ? parseInt(userData.founded) : null,
          country: 'US', // Default to US, can be made configurable
          is_active: true,
          // Set default demographics for mining/minerals industry
          target_age_min: 25,
          target_age_max: 65,
          target_gender: 'all',
          target_locations: ['USA', 'Canada', 'Australia', 'Chile'], // Major mining countries
          target_income_level: userData.category === 'consulting' ? 'high' : 'medium',
          target_lifestyle: [userData.category],
          target_interests: ['mining', 'minerals', userData.category]
        })
        .select()
        .single()
      
      if (brandError) {
        console.error('Error creating brand:', brandError)
        return NextResponse.json(
          { success: false, message: `Failed to create brand: ${brandError.message}` },
          { status: 500 }
        )
      }
      
      if (!newBrand) {
        return NextResponse.json(
          { success: false, message: 'Failed to create brand - no data returned' },
          { status: 500 }
        )
      }
      
      brandId = newBrand.brand_id
    }

    // Insert user data into the usr table with brand_id
    const { data: newUser, error: insertError } = await supabase
      .from('usr')
      .insert({
        email,
        password: 'managed_by_supabase_auth', // Don't store actual password
        name: userData.name,
        brand_id: brandId,
        role: existingBrand ? 'member' : 'admin', // First user for a new brand is admin
        is_active: true
      })
      .select()
      .single()

    if (insertError) {
      console.error('Error inserting user:', insertError)
      // If there was an error inserting into the table, but auth worked
      if (insertError.code === '23505') { // Unique violation
        return NextResponse.json(
          { success: false, message: 'Email already taken' },
          { status: 400 }
        )
      }
      return NextResponse.json(
        { success: false, message: `Failed to create user profile: ${insertError.message}` },
        { status: 500 }
      )
    }

    if (!newUser) {
      return NextResponse.json(
        { success: false, message: 'Failed to create user profile' },
        { status: 500 }
      )
    }

    return NextResponse.json({ 
      success: true, 
      message: 'Account created successfully! Please check your email to verify your account.'
    })
  } catch (error: any) {
    console.error('Signup error:', error)
    return NextResponse.json(
      { success: false, message: error.message || 'An unexpected error occurred' },
      { status: 500 }
    )
  }
}
