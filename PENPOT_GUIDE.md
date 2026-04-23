# VividMedi Penpot Design Guide

## Getting Started with Penpot

### 1. Access Penpot
- Local: http://localhost:9001
- Create account (free)
- Create new project "VividMedi"

### 2. Design System

**Colors:**
- Primary: #00d9ff (cyan)
- Secondary: #1a1f3a (dark blue)
- Dark: #0f1623 (black)
- Accent: #00a8cc
- Success: #51cf66
- Warning: #ffd43b
- Error: #ff6b6b

**Typography:**
- Font: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI"
- Heading: 28px, bold, color: #00d9ff
- Body: 14px, color: #ffffff
- Small: 12px, color: #a0b0c0

**Components to Design:**
1. Hero Section (like the AI sphere image)
   - Large gradient sphere/globe
   - Clinical AI text overlay
   - Circuit patterns background
   - Glowing accents

2. Feature Cards
   - Icon + title + description
   - Hover effects
   - Gradient borders

3. Button Styles
   - Primary (cyan gradient)
   - Secondary (outline)
   - States (hover, active, disabled)

4. Input Fields
   - Text boxes
   - Textareas
   - Selects

5. Navigation
   - Header with logo
   - User menu
   - Theme toggle

### 3. Export to HTML/CSS

**In Penpot:**
1. Design your component
2. Right-click → Export
3. Choose "HTML/CSS"
4. Copy code to static folder

**Or use Penpot CLI:**
```bash
penpot export --project "VividMedi" --output ./exported_designs/
```

### 4. Integration Steps

1. **Design in Penpot** (localhost:9001)
2. **Export components** as HTML/CSS/SVG
3. **Place in /static/components/**
4. **Import into Flask templates**
5. **Customize with Tailwind**

### 5. Design Inspiration

For VividMedi, focus on:
- Medical/clinical aesthetic (trust, professionalism)
- AI/tech elements (futuristic, intelligent)
- Dark theme (reduces eye strain, modern)
- Cyan/blue palette (healthcare standard)
- Minimalist layout (clear hierarchy)

### 6. Example: Hero Section Design

Create:
- Large sphere/orb (SVG or PNG)
- "VividMedi" text with glow effect
- Circuit board patterns (SVG)
- Floating particles (CSS animation)
- Call-to-action button

Export as SVG + CSS, then add to `templates/index.html`

## Penpot Features to Use

✅ Components library (reusable)
✅ Assets panel (icons, colors)
✅ Prototype mode (interactions)
✅ Export as code
✅ Collaboration (if needed)
✅ Version history
✅ Design tokens

## Quick Workflow

1. Open Penpot: http://localhost:9001
2. Create new file: "VividMedi UI Kit"
3. Design hero, cards, buttons
4. Export each component
5. Place exports in `/static/components/`
6. Link in Flask templates
7. Customize with Tailwind CSS

## SVG Export Tips

- Export as SVG for scalability
- Use CSS for animations
- Keep colors as hex (#00d9ff)
- Use viewBox for responsiveness

Example SVG import in HTML:
```html
<img src="{{ url_for('static', filename='components/hero.svg') }}" alt="Hero">
```

Or inline SVG:
```html
<svg viewBox="0 0 1000 600" class="w-full">
  <!-- Your design paths here -->
</svg>
```

## Next Steps

1. Start Penpot: `docker-compose up penpot`
2. Design professional hero section
3. Export as SVG/CSS
4. Create new `static/components/` folder
5. Place exports there
6. Update Flask templates to use new designs
7. Test responsiveness

Good luck! Penpot is powerful - take time to explore.
