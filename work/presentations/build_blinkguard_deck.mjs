import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SOURCE_MD = "/Users/nguyendinhhoang/Documents/DAP_GR3_SUM26/reports/web_client_side_deployment_slides_en.md";
const OUT_PPTX = "/Users/nguyendinhhoang/Documents/DAP_GR3_SUM26/outputs/web_client_side_deployment_slides_en.pptx";
const QA_DIR = "/private/tmp/codex-presentations/manual-blinkguard-web-client-side-deployment/tmp/qa";

const W = 1280;
const H = 720;
const COLORS = {
  ink: "#111111",
  muted: "#555555",
  hairline: "#D6D6D6",
  panel: "#F4F4F4",
  codePanel: "#EFEFEF",
  accent: "#1F6F6A",
  noteBg: "#F7FAF9",
};

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function stripMdInline(text) {
  return text
    .replace(/^\*\*(.*)\*\*$/u, "$1")
    .replace(/`([^`]+)`/gu, "$1")
    .replace(/\*\*/gu, "")
    .trim();
}

function parseSlides(markdown) {
  const headings = [...markdown.matchAll(/^## Slide \d+:/gmu)];
  const chunks = headings.map((match, index) => {
    const start = match.index;
    const end = index + 1 < headings.length ? headings[index + 1].index : markdown.length;
    return markdown.slice(start, end).replace(/\n---\s*$/u, "");
  });
  return chunks
    .map((chunk) => chunk.trim())
    .filter((chunk) => /^## Slide \d+:/u.test(chunk))
    .map((chunk) => {
      const lines = chunk.split(/\r?\n/u);
      const heading = lines.shift().match(/^## Slide (\d+):\s*(.+)$/u);
      const slide = {
        number: Number(heading[1]),
        section: heading[2],
        title: "",
        blocks: [],
        notes: [],
      };

      let currentList = null;
      let currentLabel = null;
      let currentCode = null;
      let inCode = false;

      function closeList() {
        if (currentList) {
          slide.blocks.push(currentList);
          currentList = null;
        }
      }

      function closeCode() {
        if (currentCode) {
          slide.blocks.push(currentCode);
          currentCode = null;
        }
      }

      for (const rawLine of lines) {
        const line = rawLine.replace(/\s+$/u, "");
        if (line.trim() === "") continue;

        if (line.startsWith("```")) {
          if (!inCode) {
            closeList();
            currentCode = { type: "code", language: line.replace(/```/u, "").trim(), lines: [] };
            inCode = true;
          } else {
            closeCode();
            inCode = false;
          }
          continue;
        }

        if (inCode) {
          currentCode.lines.push(line);
          continue;
        }

        const title = line.match(/^\*\*Title:\*\*\s*(.+)$/u);
        if (title) {
          closeList();
          slide.title = title[1].trim();
          continue;
        }

        const label = line.match(/^\*\*(.+?):\*\*\s*$/u);
        if (label) {
          closeList();
          currentLabel = label[1].trim();
          if (currentLabel === "Speaker Note") {
            continue;
          }
          slide.blocks.push({ type: "label", text: currentLabel });
          continue;
        }

        const bullet = line.match(/^(\s*)-\s+(.+)$/u);
        if (bullet) {
          const indent = Math.floor(bullet[1].length / 2);
          if (!currentList) currentList = { type: "list", items: [] };
          currentList.items.push({ indent, text: stripMdInline(bullet[2]) });
          continue;
        }

        closeList();
        const text = stripMdInline(line);
        if (currentLabel === "Speaker Note") {
          slide.notes.push(text);
        } else {
          slide.blocks.push({ type: "text", text });
        }
      }

      closeList();
      closeCode();
      return slide;
    });
}

function addText(slide, name, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text.set(text);
  box.text.style = {
    fontSize: style.fontSize ?? 20,
    color: style.color ?? COLORS.ink,
    bold: style.bold ?? false,
    typeface: "Helvetica Neue",
    wrap: "square",
    autoFit: "shrinkText",
    insets: { top: 2, right: 4, bottom: 2, left: 4 },
    ...style.extra,
  };
  return box;
}

function addPanel(slide, name, position, fill = COLORS.panel) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position,
    fill,
    line: { style: "solid", fill: COLORS.hairline, width: 1 },
  });
}

function isProcessOnly(blocks) {
  return blocks.length === 1 && blocks[0].type === "code" && blocks[0].lines.length >= 6;
}

function drawHeader(slide, data) {
  addText(slide, `slide-${data.number}-section`, `Slide ${data.number}: ${data.section}`, {
    left: 64,
    top: 34,
    width: 480,
    height: 26,
  }, { fontSize: 16, bold: true, color: COLORS.accent });
  addText(slide, `slide-${data.number}-title`, data.title, {
    left: 64,
    top: 74,
    width: 1032,
    height: 88,
  }, { fontSize: data.title.length > 42 ? 40 : 44, bold: true });
  slide.shapes.add({
    geometry: "rect",
    name: `slide-${data.number}-rule`,
    position: { left: 64, top: 168, width: 1152, height: 2 },
    fill: COLORS.ink,
    line: { style: "solid", fill: COLORS.ink, width: 0 },
  });
}

function blockTextLines(block) {
  if (block.type === "label") return [block.text];
  if (block.type === "text") return [block.text];
  if (block.type === "code") return block.lines;
  return block.items.map((item) => item.text);
}

function totalLines(blocks) {
  return blocks.reduce((sum, block) => sum + blockTextLines(block).length + (block.type === "label" ? 0.25 : 0), 0);
}

function addListBlock(slide, block, x, y, width, lineHeight, fontSize) {
  let top = y;
  for (const item of block.items) {
    const bullet = item.indent > 0 ? "–" : "•";
    addText(slide, `bullet-${Math.round(x)}-${Math.round(top)}`, bullet, {
      left: x + item.indent * 28,
      top,
      width: 20,
      height: lineHeight,
    }, { fontSize, color: COLORS.ink });
    addText(slide, `bullet-text-${Math.round(x)}-${Math.round(top)}`, item.text, {
      left: x + 28 + item.indent * 30,
      top,
      width: width - 28 - item.indent * 30,
      height: lineHeight + 4,
    }, { fontSize, color: COLORS.ink });
    top += lineHeight;
  }
  return top + 10;
}

function addCodeBlock(slide, block, x, y, width, lineHeight, fontSize) {
  const height = Math.max(lineHeight * block.lines.length + 20, 52);
  addPanel(slide, `code-panel-${Math.round(x)}-${Math.round(y)}`, { left: x, top: y, width, height }, COLORS.codePanel);
  addText(slide, `code-text-${Math.round(x)}-${Math.round(y)}`, block.lines.join("\n"), {
    left: x + 16,
    top: y + 10,
    width: width - 32,
    height: height - 20,
  }, {
    fontSize,
    color: COLORS.ink,
    extra: { typeface: "Menlo", autoFit: "shrinkText" },
  });
  return y + height + 14;
}

function addLabelBlock(slide, block, x, y, width) {
  addText(slide, `label-${block.text}-${Math.round(y)}`, block.text + ":", {
    left: x,
    top: y,
    width,
    height: 28,
  }, { fontSize: 20, bold: true, color: COLORS.accent });
  return y + 34;
}

function addTextBlock(slide, block, x, y, width, fontSize) {
  addText(slide, `text-${Math.round(y)}`, block.text, {
    left: x,
    top: y,
    width,
    height: 56,
  }, { fontSize, color: COLORS.muted });
  return y + 62;
}

function drawStandardSlide(slide, data) {
  drawHeader(slide, data);
  const contentTop = 200;
  const contentX = 86;
  const contentW = 1088;
  const lines = totalLines(data.blocks);
  const dense = lines > 13;
  const fontSize = dense ? 19 : 22;
  const codeFont = dense ? 17 : 19;
  const lineHeight = dense ? 31 : 36;
  let y = contentTop;

  for (const block of data.blocks) {
    if (block.type === "list") {
      y = addListBlock(slide, block, contentX, y, contentW, lineHeight, fontSize);
    } else if (block.type === "label") {
      y = addLabelBlock(slide, block, contentX, y + 2, contentW);
    } else if (block.type === "text") {
      y = addTextBlock(slide, block, contentX, y, contentW, fontSize);
    } else if (block.type === "code") {
      y = addCodeBlock(slide, block, contentX, y, contentW, lineHeight, codeFont);
    }
  }
}

function drawProcessSlide(slide, data) {
  drawHeader(slide, data);
  const steps = data.blocks[0].lines.filter((line) => line.trim() && line.trim() !== "↓");
  const startY = 200;
  const gap = 10;
  const boxH = 43;
  for (let i = 0; i < steps.length; i += 1) {
    const top = startY + i * (boxH + gap);
    addPanel(slide, `process-step-${i + 1}`, { left: 224, top, width: 832, height: boxH }, i % 2 === 0 ? "#F4F4F4" : "#FFFFFF");
    addText(slide, `process-step-text-${i + 1}`, steps[i].trim(), {
      left: 244,
      top: top + 7,
      width: 792,
      height: 30,
    }, { fontSize: 21, bold: i === 0 || i === steps.length - 1, color: COLORS.ink });
    if (i < steps.length - 1) {
      addText(slide, `process-arrow-${i + 1}`, "↓", {
        left: 624,
        top: top + boxH - 2,
        width: 32,
        height: 22,
      }, { fontSize: 18, bold: true, color: COLORS.accent, extra: { alignment: "center" } });
    }
  }
}

async function main() {
  await fs.mkdir(path.dirname(OUT_PPTX), { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  const markdown = await fs.readFile(SOURCE_MD, "utf8");
  const slides = parseSlides(markdown);

  if (slides.length !== 20) {
    throw new Error(`Expected 20 slides, parsed ${slides.length}`);
  }

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  for (const data of slides) {
    const slide = presentation.slides.add();
    slide.background.fill = "white";
    if (isProcessOnly(data.blocks)) {
      drawProcessSlide(slide, data);
    } else {
      drawStandardSlide(slide, data);
    }
    if (data.notes.length > 0) {
      slide.speakerNotes.textFrame.setText(data.notes);
      slide.speakerNotes.setVisible(true);
    }
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(QA_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(QA_DIR, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT_PPTX);

  const inspect = await presentation.inspect({ kind: "slide,textbox,notes", maxChars: 60000 });
  await fs.writeFile(path.join(QA_DIR, "inspect.ndjson"), inspect.ndjson);
  console.log(`Wrote ${OUT_PPTX}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
