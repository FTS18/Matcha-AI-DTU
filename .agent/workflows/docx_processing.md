---
description: DOCX creation, editing, and analysis
---

# DOCX creation, editing, and analysis

## Overview

A .docx file is a ZIP archive containing XML files.

## Quick Reference

- **Read/analyze content**: `pandoc` or `unpack` for raw XML
- **Create new document**: Use `docx-js` - see Creating New Documents below
- **Edit existing document**: Unpack → edit XML → repack - see Editing Existing Documents below

## Converting .doc to .docx

Legacy .doc files must be converted before editing:

```bash
python scripts/office/soffice.py --headless --convert-to docx document.doc
```

## Reading Content

```bash
# Text extraction with tracked changes
pandoc --track-changes=all document.docx -o output.md

# Raw XML access
python scripts/office/unpack.py document.docx unpacked/
```

## Converting to Images

```bash
python scripts/office/soffice.py --headless --convert-to pdf document.docx
pdftoppm -jpeg -r 150 document.pdf page
```

## Accepting Tracked Changes

To produce a clean document with all tracked changes accepted (requires LibreOffice):

```bash
python scripts/accept_changes.py input.docx output.docx
```

## Creating New Documents

Generate .docx files with JavaScript, then validate. Install: `npm install -g docx`

### Setup

```javascript
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  ImageRun,

  Header,
  Footer,
  AlignmentType,
  PageOrientation,
  LevelFormat,
  ExternalHyperlink,

  TableOfContents,
  HeadingLevel,
  BorderStyle,
  WidthType,
  ShadingType,

  VerticalAlign,
  PageNumber,
  PageBreak,
} = require("docx");

const doc = new Document({
  sections: [
    {
      children: [
        /* content */
      ],
    },
  ],
});
Packer.toBuffer(doc).then((buffer) => fs.writeFileSync("doc.docx", buffer));
```

### Validation

After creating the file, validate it. If validation fails, unpack, fix the XML, and repack.

```bash
python scripts/office/validate.py doc.docx
```

### Page Size

```javascript
// CRITICAL: docx-js defaults to A4, not US Letter
// Always set page size explicitly for consistent results
sections: [
  {
    properties: {
      page: {
        size: {
          width: 12240,

          // 8.5 inches in DXA

          height: 15840,

          // 11 inches in DXA
        },

        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }, // 1 inch margins
      },
    },

    children: [
      /* content */
    ],
  },
];
```

Common page sizes (DXA units, 1440 DXA = 1 inch):

- US Letter: Width 12,240, Height 15,840, Content Width 9,360
- A4 (default): Width 11,906, Height 16,838, Content Width 9,026

Landscape orientation: docx-js swaps width/height internally, so pass portrait dimensions and let it handle the swap:

```javascript
size: {

width: 12240,

// Pass SHORT edge as width

height: 15840, // Pass LONG edge as height

orientation: PageOrientation.LANDSCAPE // docx-js swaps them in the XML
},
```

### Styles (Override Built-in Headings)

```javascript
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } }, // 12pt default

    paragraphStyles: [
      // IMPORTANT: Use exact IDs to override built-in styles

      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,

        run: { size: 32, bold: true, font: "Arial" },

        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 },
      }, // outlineLevel required for TOC

      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,

        run: { size: 28, bold: true, font: "Arial" },

        paragraph: { spacing: { before: 180, after: 180 }, outlineLevel: 1 },
      },
    ],
  },

  sections: [
    {
      children: [
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          children: [new TextRun("Title")],
        }),
      ],
    },
  ],
});
```

### Lists (NEVER use unicode bullets)

```javascript
// WRONG - never manually insert bullet characters
new Paragraph({ children: [new TextRun("• Item")] }); // BAD
new Paragraph({ children: [new TextRun("\u2022 Item")] }); // BAD

// CORRECT - use numbering config with LevelFormat.BULLET
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",

        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,

            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },

      {
        reference: "numbers",

        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,

            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },

  sections: [
    {
      children: [
        new Paragraph({
          numbering: { reference: "bullets", level: 0 },

          children: [new TextRun("Bullet item")],
        }),

        new Paragraph({
          numbering: { reference: "numbers", level: 0 },

          children: [new TextRun("Numbered item")],
        }),
      ],
    },
  ],
});
```

### Tables

CRITICAL: Tables need dual widths - set both columnWidths on the table AND width on each cell.

```javascript
// CRITICAL: Always set table width for consistent rendering
// CRITICAL: Use ShadingType.CLEAR (not SOLID) to prevent black backgrounds
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

new Table({
  width: { size: 9360, type: WidthType.DXA }, // Always use DXA (percentages break in Google Docs)

  columnWidths: [4680, 4680], // Must sum to table width (DXA: 1440 = 1 inch)

  rows: [
    new TableRow({
      children: [
        new TableCell({
          borders,

          width: { size: 4680, type: WidthType.DXA }, // Also set on each cell

          shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, // CLEAR not SOLID

          margins: { top: 80, bottom: 80, left: 120, right: 120 }, // Cell padding

          children: [new Paragraph({ children: [new TextRun("Cell")] })],
        }),
      ],
    }),
  ],
});
```

### Images

```javascript
// CRITICAL: type parameter is REQUIRED
new Paragraph({
  children: [
    new ImageRun({
      type: "png", // Required: png, jpg, jpeg, gif, bmp, svg

      data: fs.readFileSync("image.png"),

      transformation: { width: 200, height: 150 },

      altText: { title: "Title", description: "Desc", name: "Name" }, // All three required
    }),
  ],
});
```

### Page Breaks

```javascript
// CRITICAL: PageBreak must be inside a Paragraph
new Paragraph({ children: [new PageBreak()] });

// Or use pageBreakBefore
new Paragraph({ pageBreakBefore: true, children: [new TextRun("New page")] });
```

### Table of Contents

```javascript
// CRITICAL: Headings must use HeadingLevel ONLY - no custom styles
new TableOfContents("Table of Contents", {
  hyperlink: true,
  headingStyleRange: "1-3",
});
```

### Headers/Footers

```javascript
sections: [
  {
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }, // 1440 = 1 inch
    },

    headers: {
      default: new Header({
        children: [new Paragraph({ children: [new TextRun("Header")] })],
      }),
    },

    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            children: [new TextRun("Page "), new TextRun({ children: [PageNumber.CURRENT] })],
          }),
        ],
      }),
    },

    children: [
      /* content */
    ],
  },
];
```

## Editing Existing Documents

Follow all 3 steps in order.

### Step 1: Unpack

```bash
python scripts/office/unpack.py document.docx unpacked/
```

Extracts XML, pretty-prints, merges adjacent runs, and converts smart quotes to XML entities.

### Step 2: Edit XML

Edit files in `unpacked/word/`. See XML Reference below for patterns. Use "Claude" as the author for tracked changes and comments by default. Use smart quotes for new content:

```xml
<w:t>Here&#x2019;s a quote: &#x201C;Hello&#x201D;</w:t>
```

Adding comments:

```bash
python scripts/comment.py unpacked/ 0 "Comment text with &amp; and &#x2019;"
```

### Step 3: Pack

```bash
python scripts/office/pack.py unpacked/ output.docx --original document.docx
```

Validates with auto-repair, condenses XML, and creates DOCX.

## XML Reference

### Tracked Changes

**Insertion**:

```xml
<w:ins w:id="1" w:author="Claude" w:date="2025-01-01T00:00:00Z">

<w:r><w:t>inserted text</w:t></w:r>
</w:ins>
```

**Deletion**:

```xml
<w:del w:id="2" w:author="Claude" w:date="2025-01-01T00:00:00Z">

<w:r><w:delText>deleted text</w:delText></w:r>
</w:del>
```

### Comments

```xml
<w:commentRangeStart w:id="0"/>
<w:del w:id="1" w:author="Claude" w:date="2025-01-01T00:00:00Z">

<w:r><w:delText>deleted</w:delText></w:r>
</w:del>
<w:r><w:t> more text</w:t></w:r>
<w:commentRangeEnd w:id="0"/>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>
```

### Images XML

1. Add image file to `word/media/`
2. Add relationship to `word/_rels/document.xml.rels`
3. Add content type to `[Content_Types].xml`
4. Reference in `document.xml`:

```xml
<w:drawing>

<wp:inline>

<wp:extent cx="914400" cy="914400"/> <!-- EMUs: 914400 = 1 inch -->

<a:graphic>

<a:graphicData uri=".../picture">

<pic:pic>

<pic:blipFill><a:blip r:embed="rId5"/></pic:blipFill>

</pic:pic>

</a:graphicData>

</a:graphic>

</wp:inline>
</w:drawing>
```
