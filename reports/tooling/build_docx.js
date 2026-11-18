// Build reports/PathSnap_Final_Technical_Report.docx from reports/tooling/report_blocks.json
// (the same content model that produced the PDF). Run from the repo root:
//   node reports/tooling/build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, Header, Footer, PageNumber,
  TableOfContents, LevelFormat,
} = require(path.join(__dirname, "node_modules", "docx"));

const ROOT = path.resolve(__dirname, "..", "..");
const blocks = JSON.parse(fs.readFileSync(path.join(__dirname, "report_blocks.json"), "utf8"));

const FONT = "Arial";
const MONO = "Courier New";
const ACC = "2A78D6";
const PAGE_W = 11906, MARGIN = 1134;           // A4, 2 cm margins (DXA)
const CONTENT_W = PAGE_W - 2 * MARGIN;          // 9638 DXA
const CONTENT_PX = Math.round(CONTENT_W / 1440 * 96);

// **bold** and `code` inline markup -> runs
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), font: MONO, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

function colWidths(b) {
  // never split a token (number, id): every column is at least as wide as its longest token
  const n = b.header.length, CH = b.small ? 64 : 74, PAD = 170;
  const rows = [b.header, ...b.rows];
  const mins = [], want = [];
  for (let i = 0; i < n; i++) {
    let tok = 1, len = 1;
    for (const r of rows) {
      const cell = String(r[i]);
      tok = Math.max(tok, ...cell.split(/\s+/).map((t) => t.length));
      len = Math.max(len, Math.min(cell.length, 40));
    }
    mins.push(tok * CH + PAD);
    want.push(len * CH + PAD);
  }
  let w = b.widths ? b.widths.map((x, i) => Math.max(x * CONTENT_W, mins[i])) : want.map((x, i) => Math.max(x, mins[i]));
  const sw = w.reduce((a, c) => a + c, 0), sm = mins.reduce((a, c) => a + c, 0);
  let dx;
  if (sw <= CONTENT_W) dx = w.map((x) => (x / sw) * CONTENT_W);
  else if (sm < CONTENT_W) {
    const extra = CONTENT_W - sm, flex = w.map((x, i) => x - mins[i]), sf = flex.reduce((a, c) => a + c, 0) || 1;
    dx = mins.map((m, i) => m + (flex[i] / sf) * extra);
  } else dx = mins.map((m) => (m / sm) * CONTENT_W);
  dx = dx.map(Math.floor);
  dx[dx.length - 1] += CONTENT_W - dx.reduce((a, c) => a + c, 0);
  return dx;
}

const border = { style: BorderStyle.SINGLE, size: 2, color: "D9D8D3" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(b) {
  const widths = colWidths(b);
  const size = b.small ? 14 : 16; // half-points
  const mk = (cells, head, shade) => new TableRow({
    tableHeader: head,
    children: cells.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: { fill: head ? ACC : (shade ? "F5F5F2" : "FFFFFF"), type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 50, bottom: 50, left: 80, right: 80 },
      children: [new Paragraph({ children: runs(String(c), { size, bold: head, color: head ? "FFFFFF" : "1B1B1A" }) })],
    })),
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(b.header, true, false), ...b.rows.map((r, j) => mk(r, false, j % 2 === 1))],
  });
}

function caption(text) {
  return new Paragraph({ spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, italics: true, size: 16, color: "52514E", font: FONT })] });
}

const children = [];
for (const b of blocks) {
  switch (b.t) {
    case "cover":
      children.push(new Paragraph({ spacing: { before: 2600 } }));
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, heading: HeadingLevel.TITLE,
        children: [new TextRun({ text: b.title, font: FONT, bold: true, size: 44, color: ACC })] }));
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 1400 },
        children: [new TextRun({ text: b.subtitle, font: FONT, size: 28 })] }));
      for (const l of b.lines) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 140 },
        children: [new TextRun({ text: l, font: FONT, size: 19, color: "52514E" })] }));
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
    case "toc":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "Contents", font: FONT })] }));
      children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
      children.push(new Paragraph({ children: [new TextRun({ text: "(Right-click the table and choose Update Field if page numbers are not shown.)", italics: true, size: 16, color: "52514E", font: FONT })] }));
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
    case "h1":
      if (b.text.startsWith("Appendix")) children.push(new Paragraph({ children: [new PageBreak()] }));
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: b.text, font: FONT })] }));
      break;
    case "h2":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: b.text, font: FONT })] }));
      break;
    case "h3":
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: b.text, font: FONT })] }));
      break;
    case "p":
      children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: runs(b.text, { size: 20 }) }));
      break;
    case "note":
      children.push(new Paragraph({ spacing: { before: 120, after: 200 },
        shading: { fill: "FDF1EA", type: ShadingType.CLEAR, color: "auto" },
        border: { top: { style: BorderStyle.SINGLE, size: 6, color: "EB6834", space: 4 },
                  bottom: { style: BorderStyle.SINGLE, size: 6, color: "EB6834", space: 4 },
                  left: { style: BorderStyle.SINGLE, size: 6, color: "EB6834", space: 4 },
                  right: { style: BorderStyle.SINGLE, size: 6, color: "EB6834", space: 4 } },
        children: runs(b.text, { size: 19 }) }));
      break;
    case "bullets":
      for (const it of b.items) children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 },
        spacing: { after: 60 }, children: runs(it, { size: 20 }) }));
      children.push(new Paragraph({ spacing: { after: 60 } }));
      break;
    case "code":
      for (const line of b.text.split("\n")) children.push(new Paragraph({ spacing: { after: 0 },
        shading: { fill: "F3F2EE", type: ShadingType.CLEAR, color: "auto" },
        children: [new TextRun({ text: line.length ? line : " ", font: MONO, size: 15 })] }));
      children.push(new Paragraph({ spacing: { after: 120 } }));
      break;
    case "refs":
      for (const r of b.items) children.push(new Paragraph({ numbering: { reference: "refs", level: 0 },
        spacing: { after: 80 }, children: runs(r, { size: 18 }) }));
      break;
    case "table":
      children.push(table(b));
      children.push(caption(b.caption));
      break;
    case "fig": {
      const file = b.path;
      const [wpx, hpx] = pngSize(file);
      let w = Math.round(CONTENT_PX * b.width);
      let h = Math.round(w * hpx / wpx);
      const hmax = 820;
      if (h > hmax) { h = hmax; w = Math.round(h * wpx / hpx); }
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
        children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h },
          altText: { title: b.caption.slice(0, 60), description: b.caption, name: path.basename(file) } })] }));
      children.push(caption(b.caption));
      break;
    }
    case "pagebreak":
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
  }
}

const doc = new Document({
  creator: "PathSnap team", title: "PathSnap: ML-Assisted GPS Map Matching and Route Reconstruction",
  description: "Final technical report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 44, bold: true, color: ACC },
        paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: ACC }, paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "1B1B1A" }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 21, bold: true, color: "1B1B1A" }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 520, hanging: 520 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1134, bottom: 1020, left: MARGIN, right: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "D9D8D3", space: 4 } },
      children: [new TextRun({ text: "PathSnap — Final Technical Report · UE24CS352A Machine Learning", size: 15, color: "52514E", font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 15, color: "52514E", font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "reports", "PathSnap_Final_Technical_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
