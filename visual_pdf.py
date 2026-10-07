"""Lossless-on-page PDF visual question extraction for the V5.25.x patch.

Mathematical glyphs/figures are rendered from the ORIGINAL PDF, never retyped,
OCR'ed or translated to LaTeX. Text extraction is used for *coordinates and answer
metadata only*. Automatic extraction fails closed if there is no unambiguous option
layout. Complex/scanned files can provide an explicit, teacher-reviewed region map.
"""
from __future__ import annotations

import io
import json
import re
import uuid
import contextvars
from pathlib import Path

import fitz
from PIL import Image, ImageChops

SECTIONS = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2, "3": 3, "4": 4}
SEC_RE = re.compile(r"^\s*(?:PHẦN|Phần)\s*(IV|III|II|I|[1-4])\b", re.I)
QUESTION_RE = re.compile(r"^\s*C\s*[âa]\s*u\s+(\d+)\s*[:.]?\s*", re.I)
SOLUTION_RE = re.compile(r"(?:Ê\s*)?Lời\s*giải\s*[:：.]?", re.I)
ANSWER_RE = re.compile(r"(?:Đáp\s*án|ĐA|Answer)\s*[:：-]\s*", re.I)
CIRCLED = {"Ⓐ":"A", "Ⓑ":"B", "Ⓒ":"C", "Ⓓ":"D", "ⓐ":"A", "ⓑ":"B", "ⓒ":"C", "ⓓ":"D"}
OPTION_RE = re.compile(r"^\s*([ABCDabcdⒶⒷⒸⒹⓐⓑⓒⓓ])\s*(?:[.)]\s*|\s+)")
PAGE_NUM_RE = re.compile(r"(?:^\s*Trang\s+\d+(?:\s*/\s*\d+)?\s*$|Mã\s*đề\s*thi.*?Trang\s+\d+(?:\s*/\s*\d+)?(?:\s*$|\b))", re.I)
SHARED_RE = re.compile(r"Sử\s*dụng(?:\s+các)?\s+thông\s*tin\s+sau\s+cho\s+Câu\s+(\d+)\s*(?:và|đến|-)\s*Câu\s+(\d+)", re.I)
DEFAULT_DPI = 180
PDF_VISUAL_TARGET_PT = 13.5
PDF_VISUAL_MAX_CSS_WIDTH = 900
_DISPLAY_SCALE = contextvars.ContextVar("visual_pdf_display_scale", default=1.0)
MAX_PAGES = 140
MAX_PIXELS_PER_PART = 24_000_000
_CREATED = contextvars.ContextVar("visual_pdf_created", default=None)

class VisualReviewRequired(ValueError):
    """The PDF needs teacher-reviewed coordinates; no silently mangled question."""


def _lines(page):
    # One PDF text line can contain multiple visual option columns. PDF char
    # coordinates come from rawdict, unlike text blocks which can merge columns.
    out=[]
    raw=page.get_text("rawdict", sort=True)
    for block in raw['blocks']:
        if 'lines' not in block: continue
        for line in block['lines']:
            chars=[]
            for sp in line['spans']:
                for ch in sp.get('chars',[]):
                    chars.append((ch['c'],fitz.Rect(ch['bbox'])))
            if not chars: continue
            text=''.join(ch for ch,_ in chars)
            if not text.strip(): continue
            bbox=fitz.Rect(line['bbox'])
            out.append({'text':text,'bbox':bbox,'chars':chars,'y':bbox.y0,'x':bbox.x0})
    return sorted(out,key=lambda v:(v['y'],v['x']))


def _events(doc):
    events=[]
    for pn,page in enumerate(doc):
        rows=_lines(page)
        question_headers=[r for r in rows if QUESTION_RE.match(r['text']) and r['x']<page.rect.width*.4]
        for row in rows:
            text=row['text']; x=row['x']; y=row['y']; bbox=row['bbox']
            # Headings and question numbers start near the page's text margin.
            sec=SEC_RE.match(text)
            if sec and x < page.rect.width*.35:
                events.append(dict(kind='section',page=pn,y=y,section=SECTIONS[sec.group(1).upper()],row=row))
                continue
            q=QUESTION_RE.match(text)
            if q and x < page.rect.width*.4:
                events.append(dict(kind='question',page=pn,y=y,number=int(q.group(1)),row=row,match=q))
                continue
            marker=ANSWER_RE.search(text)
            if marker and x < page.rect.width*.97:
                # Short-answer keys are often printed at the far right of the
                # final prompt line (e.g. ``... ?   Đáp án: 9``). Do not miss
                # them just because their x-position lies beyond 70% width.
                events.append(dict(kind='answer',page=pn,y=y,row=row,match=marker))
                continue
            # Several exam templates encode the correct choice in the final
            # solution line rather than in a separate 'Đáp án:' line.
            chosen=re.search(r'Chọn\s+đáp\s+án\s*[:：-]?\s*([ABCD])\b',text,re.I)
            if chosen and x<page.rect.width*.70:
                events.append(dict(kind='answer',page=pn,y=y,row=row,match=chosen))
                continue
            marker=SOLUTION_RE.search(text)
            if marker:
                events.append(dict(kind='solution',page=pn,y=y,row=row,match=marker))
                continue
            shared=SHARED_RE.search(text)
            if shared and x < page.rect.width*.60:
                events.append(dict(kind='shared',page=pn,y=y,row=row,start_q=int(shared.group(1)),end_q=int(shared.group(2))))
                continue
            # A label can be in an independent PDF line (4-column table) or
            # several labels can exist in one span. Search labels at start of a
            # text line; also search inline labels after 2+ spaces when necessary.
            # Columns emitted as one PDF text line. Some math fonts emit
            # ". C. (" or ". D. (" without a normal leading A/B/C/D span.
            # Only use inline labels with plausible column positions.
            for later_opt in re.finditer(r"(?<!\w)([ABCDabcd])\s*[.)](?=\s)",text):
                ci=later_opt.start(1); key=later_opt.group(1).upper()
                if ci>=len(row['chars']):continue
                xc=row['chars'][ci][1].x0
                first_label=bool(OPTION_RE.match(text))
                punct_lead=bool(re.match(r"^\s*[.)]\s*",text))
                if ci<=2 and not punct_lead:continue
                # A period after a unit symbol (e.g. '1°C.') is NOT option C.
                if ci>0 and text[ci-1] in ('°','º','℃'):
                    continue
                # V5.29.12: some PDFs split a four-column option row so the
                # C label is embedded at the end of a formula-only span, e.g.
                # "= .           C.  ".  The previous rule rejected it because
                # the span itself did not begin with A/B/C/D.  Recover such a
                # label only when it is preceded by a wide run of spaces and
                # sits in the right half of the option row; this is strict
                # enough to avoid treating ordinary prose like "C." as an
                # answer label.
                spaced_inline = bool(re.search(r"\s{4,}$", text[:ci]))
                if not ((first_label and xc-row['chars'][0][1].x0>=65) or
                        (punct_lead and xc>=page.rect.width*.38) or
                        (spaced_inline and xc>=page.rect.width*.35 and len(text.strip())<=40)):
                    continue
                endch=row['chars'][min(later_opt.end()-1,len(row['chars'])-1)][1].x1
                events.append(dict(kind='option',page=pn,y=y,x=xc,end_x=endch,
                    key=key,row=row,label_box=(xc,bbox.y0,endch,bbox.y1)))
            opt=OPTION_RE.match(text)
            if opt:
                # Some PDF fonts emit e.g. "B b" as a separate text line
                # *inside* a question containing the allele names (B, b).
                # It is not option B: a real option with an unpunctuated label
                # should not share the question-header baseline.
                unpunctuated=not re.match(r'^\s*[ABCDabcd]\s*[.)]', text)
                same_question_row=any(abs(y-qrow['y'])<1.5 and x>qrow['x']+20
                                      for qrow in question_headers)
                if unpunctuated and same_question_row:
                    continue
                letter=opt.group(1)
                key=CIRCLED.get(letter, letter.upper())
                if key in 'ABCD':
                    ci=opt.start(1); ce=opt.end()
                    left=bbox.x0
                    if ci<len(row['chars']):left=row['chars'][ci][1].x0
                    end=row['chars'][min(ce-1,len(row['chars'])-1)][1].x1
                    events.append(dict(kind='option',page=pn,y=y,x=left,end_x=end,key=key,row=row,label_box=(left,bbox.y0,end,bbox.y1)))
        # Four circled answer letters may be extracted as four *single* text
        # glyphs, with their circles drawn as PDF paths (not encoded as A.).
        # Recover only a genuine four-column row in A/B/C/D spatial order.
        bare=[row for row in _lines(page) if row['text'].strip() in ('A','B','C','D')]
        for row in bare:
            if row['text'].strip()!='A':continue
            peers=[next((candidate for candidate in bare
                if candidate['text'].strip()==key and abs(candidate['y']-row['y'])<4
                and candidate['x']>row['x']+50*(idx-1)),None)
                for idx,key in enumerate('ABCD')]
            if any(z is None for z in peers):continue
            if any(peers[k+1]['x']-peers[k]['x']<55 for k in range(3)):continue
            for z,key in zip(peers,'ABCD'):
                events.append(dict(kind='option',page=pn,y=z['y'],x=z['x'],
                    end_x=z['bbox'].x1+5,key=key,row=z,
                    label_box=(z['bbox'].x0-5,z['bbox'].y0-4,
                               z['bbox'].x1+5,z['bbox'].y1+4)))
    # PDF fonts place the four columns at slightly different vertical baselines.
    # Group event rows within 2.4pt, then sort matching option labels by x.
    events.sort(key=lambda e:(e['page'],e['y'],e.get('x',0)))
    groups=[]
    for event in events:
        if not groups or event['page']!=groups[-1][0]['page'] or event['y']-min(x['y'] for x in groups[-1])>2.4:
            groups.append([event])
        else: groups[-1].append(event)
    for group in groups:
        group.sort(key=lambda e:(0 if e['kind'] in ('section','question') else 1,
                                 e.get('x',e['row']['x'])))
    return [event for group in groups for event in group]


def _parse_answer(text,sec):
    m=ANSWER_RE.search(text)
    if not m:
        chosen=re.search(r'Chọn\s+đáp\s+án\s*[:：-]?\s*([ABCD])\b',text,re.I)
        return chosen.group(1).upper() if chosen and sec==1 else None
    tail=text[m.end():].strip()
    if sec==1:
        k=re.search(r'\b([ABCD])\b',tail,re.I)
        return k.group(1).upper() if k else None
    if sec==2:
        v=re.findall('[ĐđSsTtFf]',tail)
        return [c.upper() in ('Đ','T') for c in v[:4]] if len(v)>=4 else None
    if sec==3:
        m2=re.search(r'[-+]?\d[\d,.-]*',tail)
        return m2.group(0) if m2 else None
    return None


def _expand_rect(rect, dx=0.0, dy=0.0):
    r=fitz.Rect(rect)
    return fitz.Rect(r.x0-dx, r.y0-dy, r.x1+dx, r.y1+dy)


def _page_number_rows(page):
    rows=[]
    for row in _lines(page):
        text=str(row.get('text') or '')
        bbox=row['bbox']
        # Common Vietnamese exam footers are often split as
        # "Mã đề thi 120 - Trang 1/" + a separate final page-count glyph.
        # Detect the stable prefix as well as ordinary "Trang 1/5" rows.
        is_page_row=bool(PAGE_NUM_RE.search(text))
        if not is_page_row and bbox.y0 >= page.rect.height*0.82:
            is_page_row=bool(re.search(r'Mã\s*đề\s*thi|Trang\s+\d+',text,re.I))
        if not is_page_row:
            continue
        if bbox.y0 <= page.rect.height*0.16 or bbox.y1 >= page.rect.height*0.82:
            rows.append(row)
    return rows


def _page_content_band(page):
    top=0.0; bottom=page.rect.height
    rows=_page_number_rows(page)
    if rows:
        top_rows=[r for r in rows if r['bbox'].y0 <= page.rect.height*0.16]
        if top_rows:
            top=max(top, max(r['bbox'].y1 for r in top_rows)+3.0)
        foot=[r for r in rows if r['bbox'].y0 >= page.rect.height*0.82]
        if foot:
            # Leave a small clean gap above the printed footer.  This clips
            # detached page-count glyphs too, because they share the same band.
            bottom=min(bottom, min(r['bbox'].y0 for r in foot)-4.0)
    return top,bottom


def _trim_image_edges(img, padding=3):
    colored=ImageChops.difference(img, Image.new('RGB', img.size, 'white'))
    bbox=colored.getbbox()
    colored.close()
    if not bbox:
        return img
    l=max(0, bbox[0]-padding); t=max(0, bbox[1]-padding)
    r=min(img.width, bbox[2]+padding); b=min(img.height, bbox[3]+padding)
    if (l,t,r,b)==(0,0,img.width,img.height):
        return img
    return img.crop((l,t,r,b))


def _image_blocks(page):
    blocks=[]
    try:
        info=page.get_text('dict')
    except Exception:
        return blocks
    for block in info.get('blocks', []):
        if block.get('type')!=1:
            continue
        bbox=block.get('bbox')
        if not bbox:
            continue
        rect=fitz.Rect(bbox)
        if rect.width < 6 or rect.height < 6:
            continue
        blocks.append(rect)
    return blocks


def _find_ink_seam(page, x0, x1, y0, y1, fallback):
    """Find a low-ink horizontal separator between two visual option rows.

    PDF text bboxes for stacked fractions overlap vertically.  A geometric
    midpoint can therefore cut off the denominator of the upper choice or the
    numerator of the lower choice.  Render only the option column and choose
    the whitest raster row near the middle of the gap.
    """
    try:
        x0=max(0.0,float(x0)); x1=min(float(page.rect.width),float(x1))
        y0=max(0.0,float(y0)); y1=min(float(page.rect.height),float(y1))
        if x1-x0 < 24 or y1-y0 < 5:
            return float(fallback)
        scale=2.0
        clip=fitz.Rect(x0,y0,x1,y1)
        pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=clip,alpha=False)
        mode='RGB' if pix.n>=3 else 'L'
        img=Image.frombytes(mode,[pix.width,pix.height],pix.samples)
        if mode!='L':
            img=img.convert('L')
        # Ignore the option label margin; it can contain a large A/B/C/D glyph.
        left=max(0,int(img.width*0.10)); right=max(left+1,img.width)
        # Search the middle 70% only.  This avoids choosing the white border at
        # the very top/bottom of the clip rather than the true row separator.
        lo=max(0,int(img.height*0.15)); hi=min(img.height,int(img.height*0.85))
        if hi<=lo:
            return float(fallback)
        best=None
        for yy in range(lo,hi):
            row=img.crop((left,yy,right,yy+1))
            # Count non-white-ish pixels. Anti-aliased math glyphs are <245.
            ink=sum(1 for v in row.getdata() if v < 245)
            # Prefer rows close to the requested fallback when ink ties.
            pdf_y=y0+(yy+0.5)/scale
            score=(ink,abs(pdf_y-float(fallback)))
            if best is None or score<best[0]:
                best=(score,pdf_y)
        return best[1] if best else float(fallback)
    except Exception:
        return float(fallback)


def _row_groups_vertical_regions(page, rows, left, right, top=None, bottom=None, gap=8.0, pad_y=1.0):
    rows=sorted(rows, key=lambda row: row['bbox'].y0)
    if not rows or right-left < 20:
        return []
    groups=[[rows[0]]]
    for row in rows[1:]:
        prev=groups[-1][-1]
        if row['bbox'].y0 - prev['bbox'].y1 <= gap:
            groups[-1].append(row)
        else:
            groups.append([row])
    rects=[]
    for group in groups:
        y0=max(0.0, min(row['bbox'].y0 for row in group)-pad_y)
        y1=min(page.rect.height, max(row['bbox'].y1 for row in group)+pad_y)
        if top is not None:
            y0=max(y0, top)
        if bottom is not None:
            y1=min(y1, bottom)
        if right-left >= 20 and y1-y0 >= 6:
            rects.append([left, y0, right, y1])
    return rects


def _question_label_box(event):
    """Return only the printed Câu n prefix bbox, never the whole question row."""
    try:
        row=event.get('row') or {}
        match=event.get('match')
        chars=row.get('chars') or []
        end=int(match.end()) if match is not None else 0
        if end<=0 or not chars:
            return None
        boxes=[fitz.Rect(box) for _,box in chars[:min(end,len(chars))]]
        if not boxes:
            return None
        rect=boxes[0]
        for box in boxes[1:]:
            rect |= box
        return [rect.x0-1.0,rect.y0-1.0,rect.x1+2.0,rect.y1+1.0]
    except Exception:
        return None


def _clip_part(page,rect,media_dir,blocks,dpi=DEFAULT_DPI,blank=None,tight=False):
    bbox=fitz.Rect(rect) & page.rect
    if bbox.is_empty or bbox.width < 2 or bbox.height < 2:
        raise VisualReviewRequired('Vùng cắt PDF rỗng; vui lòng kiểm tra vùng cắt trước khi xuất bản.')
    scale=dpi/72
    if bbox.width*scale*bbox.height*scale>MAX_PIXELS_PER_PART:
        raise VisualReviewRequired('Vùng cắt PDF quá lớn; chia thành nhiều phần hoặc chọn vùng nhỏ hơn.')
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=bbox,alpha=False)
    if pix.width*pix.height > MAX_PIXELS_PER_PART:
        raise VisualReviewRequired('Ảnh câu hỏi quá lớn, cần chia thành các vùng cắt ngắn hơn.')
    img=Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
    # Remove ONLY the printed dynamic label. Preserve the rest of the original
    # raster, including all math, tables, figures and typography.
    if blank:
        from PIL import ImageDraw
        blanks=blank if isinstance(blank,(list,tuple)) and blank and isinstance(blank[0],(list,tuple,fitz.Rect)) else [blank]
        factor=dpi/72
        drawer=ImageDraw.Draw(img)
        for item in blanks:
            label=fitz.Rect(item) & bbox
            if label.is_empty:
                continue
            box=((label.x0-bbox.x0)*factor,(label.y0-bbox.y0)*factor,
                 (label.x1-bbox.x0)*factor,(label.y1-bbox.y0)*factor)
            drawer.rectangle(box,fill='white')
    if tight:
        # Crop each *answer image* to its own actual printed ink instead of
        # retaining a full PDF page-width strip, which made A seem to include
        # B/C/D and long blank areas.
        colored=ImageChops.difference(img,Image.new('RGB',img.size,'white'))
        ink=colored.getbbox()
        colored.close()
        if not ink:
            raise VisualReviewRequired('Vùng phương án không chứa nội dung nhìn thấy được.')
        padding=max(2,round(dpi/72))
        clipped=(max(0,ink[0]-padding),max(0,ink[1]-padding),
                 min(img.width,ink[2]+padding),min(img.height,ink[3]+padding))
        img=img.crop(clipped)
    else:
        # V5.29.13 EXACT GEOMETRY: stem/figure crops keep the PDF rectangle
        # exactly as selected.  Do not trim white edges here: trimming changes
        # the apparent horizontal position of charts/formulas relative to the
        # original page and can also pull page-footer ink into multi-page joins.
        # Option crops still use tight=True and are cropped independently.
        pass
    media_dir.mkdir(parents=True,exist_ok=True)
    filename='pdf_visual_'+uuid.uuid4().hex+'.png'
    dest=media_dir/filename
    img.save(dest,optimize=True)
    created=_CREATED.get()
    if created is not None:created.append(dest)
    display_scale=max(1.0,float(_DISPLAY_SCALE.get() or 1.0))
    width_px=min(PDF_VISUAL_MAX_CSS_WIDTH,round(img.width*96/dpi*display_scale))
    height_px=round(img.height*width_px/img.width)
    blocks.append({'type':'image','role':'figure','visual_pdf':True,
       'url':'/question-media/'+filename,'display_width_px':width_px,
       'display_height_px':height_px,
       'word_width_cm':min(16.0,max(1.0,img.width/ (dpi/72) *2.54/72))})
    return dest


def _merge_parts(files,media_dir,dpi=DEFAULT_DPI):
    if len(files)<2:return files
    images=[]
    try:
        for p in files:
            with Image.open(p) as source:
                images.append(source.convert('RGB'))
        width=max(i.width for i in images); height=sum(i.height for i in images)
        if width*height>MAX_PIXELS_PER_PART:
            raise VisualReviewRequired('Một câu trải nhiều trang quá dài; cần chia nhỏ hoặc xem lại vùng cắt.')
        result=Image.new('RGB',(width,height),'white'); y=0
        try:
            for img in images:
                result.paste(img,(0,y));y+=img.height
            result.save(files[0],optimize=True)
        finally:
            result.close()
        for extra in files[1:]: extra.unlink(missing_ok=True)
        return files[:1]
    finally:
        for img in images:img.close()


def _ensure_readable_stem_display(block, path, slot, dpi=DEFAULT_DPI):
    """Keep short one-line visual stems readable without distorting ratio.

    V5.27.2 could render a one-line essay stem at only ~13 CSS px high after
    converting 180-DPI raster pixels back to CSS pixels.  Browsers then showed
    only a thin sliver of the question above the answer card.  Scale only
    ``stem_text`` blocks up proportionally, capped at the normal 730 px width.
    """
    if slot != 'stem_text' or not block or not path:
        return block
    try:
        current_h=float(block.get('display_height_px') or 0)
        if current_h >= 18:
            return block
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                return block
            target_h=18.0
            scale=target_h / max(1.0,current_h or (image.height*96/dpi))
            current_w=float(block.get('display_width_px') or (image.width*96/dpi))
            width=min(PDF_VISUAL_MAX_CSS_WIDTH, max(1, round(current_w*scale)))
            height=max(18, round(image.height*width/image.width))
            block['display_width_px']=width
            block['display_height_px']=height
    except Exception:
        pass
    return block


def _render_rects(doc,items,media_dir,dpi=DEFAULT_DPI,tight=False):
    # Each question/option/solution is one logical image even when PDF spans pages.
    paths=[]; blocks=[]
    for item in items:
        pn=int(item['page'])
        if pn<0 or pn>=len(doc):raise VisualReviewRequired('Số trang cắt không có trong PDF.')
        _clip_part(doc[pn],item['rect'],media_dir,blocks,dpi=dpi,blank=item.get('blank'),tight=item.get('tight', tight))
        paths.append(media_dir/Path(blocks[-1]['url']).name)
    # A question may have two semantic parts: text stem and independent chart.
    # Merge consecutive strips inside each part, but never merge a figure into
    # the text screenshot (that made charts far away / embedded in options).
    if len(paths)>1:
        grouped=[]
        for j,item in enumerate(items):
            slot=item.get('slot')
            if not grouped or grouped[-1][0]!=slot:
                grouped.append((slot,[j]))
            else:
                grouped[-1][1].append(j)
        merged=[]
        for slot, indexes in grouped:
            first=indexes[0]
            if len(indexes)>1:
                _merge_parts([paths[j] for j in indexes],media_dir,dpi)
                with Image.open(paths[first]) as image:
                    blocks[first]['display_width_px']=min(PDF_VISUAL_MAX_CSS_WIDTH,round(image.width*96/dpi*max(1.0,float(_DISPLAY_SCALE.get() or 1.0))))
                    blocks[first]['display_height_px']=round(image.height*blocks[first]['display_width_px']/image.width)
            block=blocks[first]
            if slot:
                block['slot']=slot
            _ensure_readable_stem_display(block, paths[first], slot, dpi)
            merged.append(block)
        return merged
    for j,(block,item) in enumerate(zip(blocks,items)):
        slot=item.get('slot')
        if slot:
            block['slot']=slot
        _ensure_readable_stem_display(block, paths[j] if j < len(paths) else None, slot, dpi)
    return blocks


def _manual_questions(doc,media_dir,manifest,render=None):
    if isinstance(manifest,str):manifest=json.loads(manifest)
    if not isinstance(manifest,dict) or not isinstance(manifest.get('questions'),list) or not manifest['questions']:
        raise VisualReviewRequired('Bản đồ vùng cắt phải có danh sách questions không rỗng.')
    result=[]; counters={1:0,2:0,3:0,4:0}
    render=render or (lambda regions:_render_rects(doc,regions,media_dir))
    for idx,src in enumerate(manifest['questions'],1):
        sec=int(src.get('section',0))
        if sec not in counters:raise VisualReviewRequired('Phần thi phải là 1, 2, 3 hoặc 4.')
        counters[sec]+=1
        stem=render(src.get('stem') or [])
        if not stem:raise VisualReviewRequired(f'Câu {idx}: thiếu ảnh đề bài.')
        options=[]
        if sec in (1,2):
            opts=src.get('options') or {}
            for key in 'ABCD':
                if not opts.get(key):raise VisualReviewRequired(f'Câu {idx}: thiếu vùng ảnh phương án {key}.')
                options.append({'key':key,'content':render(opts[key])})
        solution=render(src.get('solution') or [])
        correct=str(src.get('correct_option') or '').upper() if sec==1 else None
        if sec==1 and correct not in ('A','B','C','D'):
            raise VisualReviewRequired(f'Câu {idx}: phải nhập đáp án đúng A/B/C/D trước khi lưu.')
        tf=src.get('tf_answers') or []
        if sec==2 and (len(tf)!=4 or any(type(v) is not bool for v in tf)):
            raise VisualReviewRequired(f'Câu {idx}: phải nhập đủ bốn giá trị đúng/sai.')
        if sec==3 and not re.fullmatch(r'-?\d+(?:,\d+)?',str(src.get('short_answer') or '').strip()):
            raise VisualReviewRequired(f'Câu {idx}: đáp án ngắn chỉ được gồm số, dấu trừ đầu hoặc dấu phẩy thập phân.')
        result.append({'original_order':idx,'section_order':counters[sec],
           'section':sec,'qtype':{1:'mcq',2:'tf',3:'short',4:'essay'}[sec],
           'stem':stem,'options':options,'solution':solution,
           'correct_option':correct or None,'tf_answers':tf,
           'short_answer':str(src.get('short_answer') or '').strip() if sec==3 else None})
    return result


def _automatic_questions(doc,media_dir):
    events=_events(doc)
    questions=[]; shared_markers=[]; current_section=1; previous_number={}
    for i,e in enumerate(events):
        if e['kind']=='section':
            current_section=e['section']
            continue
        if e['kind']=='shared':
            shared_markers.append((i,e,current_section))
            continue
        if e['kind']=='question':
            # An explanation may contain a printed "Câu 3" referring back to
            # the same question. It is NOT a new question and must not split
            # the current answer/solution region or create duplicate exams.
            if previous_number.get(current_section)==e['number']:
                continue
            previous_number[current_section]=e['number']
            questions.append((i,e,current_section))
    if not questions:raise VisualReviewRequired('PDF không có lớp văn bản nhận diện câu hỏi; không thể tự động chia chính xác bằng ảnh đơn thuần. Vui lòng dùng PDF có lớp văn bản hoặc bản DOCX gốc.')
    first_question_for_section_number={}
    for qi,(idx,qe,qsec) in enumerate(questions):
        first_question_for_section_number.setdefault((qsec,qe['number']), (qi,idx,qe))
    result=[]; counters={1:0,2:0,3:0,4:0}
    # Safety: no guesswork for strange numeric ordering/unknown regions.
    for index,(start,e,sec) in enumerate(questions):
        next_e=events[questions[index+1][0]] if index+1<len(questions) else None
        if next_e and next_e['page']==e['page'] and next_e['y']<=e['y']:
            raise VisualReviewRequired('Phát hiện hai câu nằm ngang; cần chỉnh vùng cắt thủ công.')
        group=[x for x in events[start+1:(questions[index+1][0] if index+1<len(questions) else len(events))]
               if x['kind']!='section']
        for ss in events[start+1:(questions[index+1][0] if index+1<len(questions) else len(events))]:
            if ss['kind']=='section':
                # Last question in preceding section can be closed by section header.
                group=[x for x in group if (x['page'],x['y'])<(ss['page'],ss['y'])]
                next_e=ss
                break
        # PDF typesetting sometimes shifts the labels in different columns by
        # 4–10pt (superscripts, stacked fractions). Their raw PDF line order can
        # therefore be B,A,D,C, even though the *visual* order is A,B,C,D.
        # Choose each printed label once by key, then group it spatially.
        # In true/false PDFs the detailed solution repeats a), b), c), d).
        # Never use those explanatory labels as selectable options: select
        # only option markers occurring BEFORE the first solution/answer.
        first_explanation=min(((z['page'],z['row']['bbox'].y0) for z in group
                     if z['kind'] in ('solution','answer')), default=None)
        label_events=[candidate for candidate in group
                      if candidate['kind']=='option' and
                      (first_explanation is None or
                       (candidate['page'],candidate['row']['bbox'].y0)<first_explanation)] if sec in (1,2) else []
        if sec==2 and label_events:
            # V5.29.5: in true/false stems, variable definitions often begin
            # with uppercase 'A ', 'B ', 'C ' (e.g. 'B là biến cố ...').
            # They are mathematical/prose variables, NOT a/b/c/d statements.
            # Prefer the explicit lowercase punctuated labels a), b), c), d).
            explicit_tf=[candidate for candidate in label_events
                if re.match(r'^\s*[abcd]\s*[.)]', candidate['row']['text'])]
            explicit_keys={candidate['key'] for candidate in explicit_tf}
            if explicit_keys==set('ABCD'):
                label_events=explicit_tf
        # Option-like "a = 1" inside a short-answer solution is mathematical
        # notation, not a multiple-choice choice. In sections III/IV no option
        # crops are made; otherwise solutions could be misplaced in the stem.
        opt_by_key={}
        for key in 'ABCD':
            matches=[candidate for candidate in label_events if candidate['key']==key]
            if matches:
                if sec==2:
                    # True/false templates may use either a) / b) / ... or
                    # a. / b. / ... labels. Do not reject the real statement
                    # rows just because the punctuation style differs.
                    matches=[z for z in matches if
                             re.match(r'^\s*'+key.lower()+r'(?:[.)]\s*|\s+)', z['row']['text'].lower())]
                if matches:opt_by_key[key]=matches[0]
        # The label D can be emitted as a lone glyph when the answer is a
        # stacked fraction. Recover it only alongside A/B/C in the same
        # four-column row, and only at the expected right-hand position.
        if sec==1 and set(opt_by_key)==set('ABC'):
            peers=list(opt_by_key.values())
            for candidate in _lines(doc[peers[0]['page']]):
                if candidate['text'].strip()!='D':continue
                if candidate['x']<=max(z['x'] for z in peers)+35:continue
                if min(abs(candidate['y']-z['y']) for z in peers)>12:continue
                label_rect=candidate['bbox']
                opt_by_key['D']=dict(kind='option',page=peers[0]['page'],
                    y=candidate['y'],x=candidate['x'],end_x=label_rect.x1,
                    key='D',row=candidate,
                    label_box=(label_rect.x0,label_rect.y0,label_rect.x1,label_rect.y1))
                break
        # D. can also be appended to the end of a long PDF formula fragment,
        # e.g. "v.u. D. (". Recover it only if A/B/C are already positioned
        # in columns in the very same answer row.
        if sec==1 and set(opt_by_key)==set('ABC'):
            peers=list(opt_by_key.values());pg=doc[peers[0]['page']]
            for row in _lines(pg):
                if min(abs(row['y']-z['y']) for z in peers)>12:continue
                for m in re.finditer(r'(?<!\w)D\s*[.)](?=\s)',row['text']):
                    if m.start()>=len(row['chars']):continue
                    left=row['chars'][m.start()][1].x0
                    if left<=max(z['x'] for z in peers)+40:continue
                    right=row['chars'][min(m.end()-1,len(row['chars'])-1)][1].x1
                    opt_by_key['D']=dict(kind='option',page=peers[0]['page'],y=row['y'],
                       x=left,end_x=right,key='D',row=row,
                       label_box=(left,row['bbox'].y0,right,row['bbox'].y1))
                    break
                if 'D' in opt_by_key:break
        opts=[opt_by_key[key] for key in 'ABCD' if key in opt_by_key]
        solutions=[x for x in group if x['kind']=='solution']
        answers=[x for x in group if x['kind']=='answer']
        if sec in (1,2):
            optkeys=[x['key'] for x in opts]
            if optkeys!=list('ABCD'):
                raise VisualReviewRequired(f'Phần {sec} câu {e["number"]}: không nhận diện được chính xác A/B/C/D ({optkeys}); không thể tự động tách an toàn. Kiểm tra bố cục PDF gốc.')
        # Reject overlapped answer/solution/option markers on same PDF row.
        cut_events=opts+solutions+answers
        for x in cut_events:
            if x['page']==e['page'] and abs(x['y']-e['y'])<1.5:
                raise VisualReviewRequired(f'Câu {e["number"]}: nhãn câu trùng hàng với phương án/lời giải.')
        first_opt=min(opts,key=lambda z:(z['page'],z['row']['bbox'].y0)) if opts else None
        # Some circled options are rendered as vertical systems/fractions. The
        # printed A/B/C/D is vertically centered between numerator and
        # denominator; the option *starts well ABOVE its letter*.
        circled_options=(len(opts)==4 and all(
            z['row']['text'].strip() in ('A','B','C','D') for z in opts))
        shared_in_group=[x for x in group if x.get('kind')=='shared']
        first_shared=shared_in_group[0] if shared_in_group else None
        first_sol=solutions[0] if solutions else None
        first_ans=answers[0] if answers else None
        if first_opt and any((m['page'],m['y'])<(first_opt['page'],first_opt['y']) for m in solutions+answers):
            raise VisualReviewRequired(f'Câu {e["number"]}: lời giải/đáp án nằm trước phương án; cần duyệt vùng cắt.')
        if sec in (1,2) and not first_opt:raise VisualReviewRequired(f'Câu {e["number"]}: thiếu phương án.')
        if any(a['page']==o['page'] and abs(a['y']-o['y'])<9 for a in answers for o in opts):
            raise VisualReviewRequired(f'Câu {e["number"]}: đáp án in cùng hàng với phương án; cần cắt thủ công.')
        # Limit each question to the next heading. Some PDF math glyphs of the
        # *next* question (for example a numerator) are positioned several
        # points ABOVE its printed 'Câu ...' baseline, so using next_e.y alone
        # leaks those glyphs into the current question. Pull the boundary up to
        # the earliest nearby row belonging to the next question's first line.
        limit=(next_e['page'],next_e['y']) if next_e else (len(doc)-1,doc[-1].rect.height-18)
        if next_e and next_e['page']>=0:
            next_page=doc[next_e['page']]
            next_top=next_e['row']['bbox'].y0
            nearby=[]
            for row in _lines(next_page):
                if next_top-9.0 <= row['bbox'].y0 <= next_top+4.0:
                    nearby.append(row['bbox'].y0)
            if nearby:
                limit=(next_e['page'], max(0.0,min(nearby)-1.2))
        page_lines_cache={}
        def page_lines(pn):
            if pn not in page_lines_cache:
                page_lines_cache[pn]=_lines(doc[pn])
            return page_lines_cache[pn]
        shared_for_question=None
        for shared_index, shared_event, shared_section in shared_markers:
            if shared_section!=sec:
                continue
            if not (shared_event['start_q'] <= e['number'] <= shared_event['end_q']):
                continue
            lookup=first_question_for_section_number.get((sec, shared_event['start_q']))
            if not lookup:
                continue
            target_qi, target_event_index, target_event = lookup
            if target_event_index <= shared_index:
                continue
            shared_for_question=(shared_index, shared_event, target_event_index, target_event)
        def previous_stem_line(pn, opt_y):
            # A tall formula may start ABOVE its option label's text bbox.
            # Compare starts, NOT the preceding line's end: otherwise a long
            # question's final line leaks into A/B/C/D (reported in V5.25.7).
            start_y=e['row']['bbox'].y0 if pn==e['page'] else 32
            # Include the first line of this very question, but not older
            # question/solution text. Never pick an option header as stem.
            candidates=[row for row in page_lines(pn)
                if start_y-.25<=row['bbox'].y0<opt_y-4.0
                and row['bbox'].x0<=e['row']['bbox'].x0+100.0
                and len(row['text'].strip())>=4
                # Reject rotated/tall page-side labels. Their y0 may sit just
                # above an option while y1 extends hundreds of points below,
                # which previously pushed the option crop beneath the page.
                and row['bbox'].height<=max(32.0,row['bbox'].width*1.8)
                and not OPTION_RE.match(row['text'])
                and not SOLUTION_RE.search(row['text'])
                and not ANSWER_RE.search(row['text'])]
            if pn==e['page'] and e['row'] not in candidates:
                candidates.append(e['row'])
            return max(candidates,key=lambda row:row['bbox'].y0) if candidates else None
        def strips(start_marker,end_marker,blank=None, top_pad=0.8, bottom_pad=0.6, first_x0=None):
            sp,sy=start_marker;ep,ey=end_marker
            if (sp,sy)>=(ep,ey):return []
            chunks=[]
            for pn in range(sp,ep+1):
                page=doc[pn]
                content_top, content_bottom = _page_content_band(page)
                top=max(content_top, sy-top_pad) if pn==sp else content_top
                bottom=min(content_bottom, ey-bottom_pad) if pn==ep else content_bottom
                if bottom-top<2:
                    continue
                x0=(first_x0 if pn==sp and first_x0 is not None else max(0,page.rect.width*.045))
                blanks=[]
                for row in _page_number_rows(page):
                    rect = _expand_rect(row['bbox'], dx=3.0, dy=2.0) & page.rect
                    if rect.y1 <= top or rect.y0 >= bottom:
                        continue
                    blanks.append([rect.x0,rect.y0,rect.x1,rect.y1])
                if blank is not None and pn==sp:
                    if isinstance(blank,(list,tuple)) and blank and isinstance(blank[0],(int,float)):
                        blanks.append(blank)
                    elif isinstance(blank,list):
                        blanks.extend(blank)
                    else:
                        blanks.append(blank)
                entry={'page':pn,'rect':[x0,top,min(page.rect.width,page.rect.width*.955),bottom]}
                if blanks:
                    entry['blank']=blanks
                chunks.append(entry)
            return chunks
        # Determine the true last text line of the question instead of using
        # the printed option label as the cutoff. In many math/physics PDFs the
        # numerator of option A is positioned ABOVE the A label; using A.y0 as
        # the stem boundary therefore copied option formula fragments into the
        # question (e.g. Math Q5/Q9/Q11 and Physics Q16).
        stem_text_rows=[]
        def _wordish(row):
            return bool(re.search(r'[A-Za-zÀ-ỹĐđ]{2,}', row['text']))
        if first_opt:
            for pn in range(e['page'], first_opt['page']+1):
                page=doc[pn]
                content_top,content_bottom=_page_content_band(page)
                low=(e['row']['bbox'].y0-0.5 if pn==e['page'] else content_top)
                high=(first_opt['row']['bbox'].y0-0.45 if pn==first_opt['page'] else content_bottom)
                page_rows=[]
                for row in page_lines(pn):
                    if not (low <= row['bbox'].y0 < high):
                        continue
                    if ANSWER_RE.search(row['text']) or SOLUTION_RE.search(row['text']) or SEC_RE.match(row['text']):
                        continue
                    # Never treat an actual option marker as question text.
                    if OPTION_RE.match(row['text']):
                        continue
                    if QUESTION_RE.match(row['text']) or _wordish(row):
                        # V5.29.0: stacked-fraction / trig answer glyphs may
                        # expose short word-like fragments (sin, cos, log, ...)
                        # a few points ABOVE the printed A/B/C/D label. They
                        # belong to the option crop, not to the stem. Reject a
                        # short detached token very near the first option row
                        # unless it starts in the normal left text column.
                        if (pn==first_opt['page']
                                and first_opt['row']['bbox'].y0-row['bbox'].y0 < 10.0
                                and len(row['text'].strip()) <= 14
                                and row['bbox'].x0 > e['row']['bbox'].x0+40.0
                                and not QUESTION_RE.match(row['text'])):
                            continue
                        page_rows.append(row)
                        continue
                    # Formula-only lines can belong to the stem, but only when
                    # they sit close to already accepted question text and are
                    # not immediately above the first option label.
                    if page_rows:
                        gap=row['bbox'].y0-page_rows[-1]['bbox'].y1
                        dist=(first_opt['row']['bbox'].y0-row['bbox'].y1
                              if pn==first_opt['page'] else 999)
                        if gap <= 5.5 and dist >= 8.0:
                            page_rows.append(row)
                stem_text_rows.extend(page_rows)
            if not stem_text_rows:
                stem_text_rows=[e['row']]
            last_text=max(stem_text_rows,key=lambda r:(next((pn for pn in range(e['page'],first_opt['page']+1) if r in page_lines(pn)),e['page']),r['bbox'].y0))
            # Resolve the page containing the last accepted text line.
            last_page=e['page']
            for pn in range(e['page'],first_opt['page']+1):
                if any(r is last_text for r in page_lines(pn)):
                    last_page=pn; break
            # V5.29.0 PDF EXACT CUT: stop immediately after the last verified
            # stem text bbox.  A larger +4pt pad can capture the top strokes of
            # stacked-fraction options (radicals/fraction bars) that are drawn
            # several points above their A/B/C/D labels.  The PDF bbox already
            # includes glyph ascenders/descenders, so 0.7pt is enough anti-alias
            # safety without mixing the first option into the question crop.
            text_stem_end=(last_page,min(last_text['bbox'].y1+0.7,(first_opt['row']['bbox'].y0-1.0) if first_opt and first_opt['page']==last_page else doc[last_page].rect.height-24))
        else:
            boundary=(first_shared['page'], first_shared['row']['bbox'].y0-0.35) if first_shared else None
            text_stem_end=(boundary if boundary else ((first_sol['page'],first_sol['row']['bbox'].y0-0.35) if first_sol else (
                (first_ans['page'],first_ans['row']['bbox'].y0-0.35) if first_ans else limit)))
            # A figure belonging to the NEXT short-answer question can begin a
            # few points before the current answer line. Do not let its top
            # edge leak into the current question (Physics Part III Q1).
            if sec in (3,4) and first_ans and first_ans['page']==e['page']:
                crossing=[r for r in _image_blocks(doc[e['page']])
                    if r.y0 >= e['row']['bbox'].y0-1
                    and r.y0 < first_ans['row']['bbox'].y0
                    and r.y1 > first_ans['row']['bbox'].y0+12]
                if crossing:
                    text_stem_end=(e['page'],min(text_stem_end[1],min(r.y0 for r in crossing)-1.0))

        # Detect charts / geometry drawings that belong to the stem. In many
        # school PDFs the figure is drawn on the right side of the question and
        # extends lower than A/B/C/D. We keep the text stem limited to the
        # first option marker and append the figure as an extra stem region.
        stem_graph_left=None
        stem_graph_rect=None
        if opts:
            stem_page=e['page']
            page=doc[stem_page]
            same_page_opts=[o for o in opts if o['page']==stem_page]
            if same_page_opts:
                option_right=max(o['row']['bbox'].x1 for o in same_page_opts)
                stem_left_rows=[row for row in page_lines(stem_page)
                    if row['bbox'].y0 >= e['row']['bbox'].y0-1.0
                    and row['bbox'].y0 < (first_opt['row']['bbox'].y0 if first_opt and first_opt['page']==stem_page else page.rect.height)
                    and row['bbox'].x0 < page.rect.width*.55
                    and not OPTION_RE.match(row['text'])
                    and not ANSWER_RE.search(row['text'])
                    and not SOLUTION_RE.search(row['text'])]
                stem_text_right=max((row['bbox'].x1 for row in stem_left_rows), default=e['row']['bbox'].x1)
                figure_left_threshold=max(e['row']['bbox'].x0+115.0, stem_text_right+7.0)
                search_bottom=page.rect.height-24
                for marker_item in (first_sol, first_ans):
                    if marker_item and marker_item['page']==stem_page and marker_item['row']['bbox'].y0 > e['row']['bbox'].y1 + 8:
                        search_bottom=min(search_bottom, marker_item['row']['bbox'].y0-1.0)
                draw_rects=[]
                local_band=max(40.0, search_bottom - e['row']['bbox'].y0)
                for rect in [d.get('rect') for d in page.get_drawings()] + _image_blocks(page):
                    if rect is None:
                        continue
                    rect=fitz.Rect(rect)
                    if max(rect.width, rect.height) < 6:
                        continue
                    if rect.width > page.rect.width*0.92 and rect.height > local_band*0.90:
                        continue
                    # Ignore clipping rectangles that are far taller than the
                    # local question band; they otherwise pull in neighbour
                    # content from other questions on the same page.
                    if rect.height > local_band*1.35 and rect.width > 18:
                        continue
                    if rect.y0 < e['row']['bbox'].y0-40 and rect.y1 > search_bottom+20:
                        continue
                    if rect.y1 < e['row']['bbox'].y0-2 or rect.y0 > search_bottom:
                        continue
                    if (first_opt and first_opt['page']==stem_page
                            and rect.y0>=first_opt['row']['bbox'].y0-18.0):
                        # A formula bar / radical belonging to answer C or D
                        # is not a diagram attached to the stem. Treating it
                        # as one truncated the last words of the question.
                        continue
                    if rect.x0 >= figure_left_threshold:
                        draw_rects.append(fitz.Rect(rect.x0-1.2, rect.y0-1.2, rect.x1+1.2, rect.y1+1.2))
                if draw_rects:
                    union=fitz.Rect(draw_rects[0])
                    for candidate in draw_rects[1:]:
                        union |= candidate
                    # Tiny radical bars / fraction strokes are mathematical
                    # glyphs, not figures. Require a substantial 2-D footprint
                    # before separating a right-side figure from the stem.
                    if union.width >= 65 and union.height >= 38:
                        stem_graph_left=union.x0
                        stem_graph_rect=(stem_page,
                            [max(page.rect.width*.045, union.x0-2.5),
                             max(e['row']['bbox'].y0, union.y0-1.0),
                             min(page.rect.width, union.x1+2.5),
                             min(page.rect.height, union.y1+1.5)])
        stem_end=text_stem_end
        qchar=e['row']['chars'][0][1] if e['row'].get('chars') else e['row']['bbox']
        stem_first_x0=max(0.0, qchar.x0-2.5)
        # The section header immediately preceding a first question has a
        # PDF text bbox that overlaps its next line. Trim just the preceding
        # heading's descenders rather than including them in the question.
        heading_just_above=any(z['kind']=='section' and z['page']==e['page']
                     and 0<e['row']['bbox'].y0-z['row']['bbox'].y0<19
                     for z in events[max(0,start-5):start])
        # Start a little ABOVE the printed 'Câu ...' row so numerators, roots
        # and superscripts that sit above its baseline are not clipped. Clamp
        # that headroom to the previous semantic boundary (section/answer/etc.)
        # so the previous question never leaks in.
        earlier=[ev for ev in events[:start]
                 if ev['page']==e['page'] and ev['kind'] in ('section','question','answer','solution')
                 and ev['row']['bbox'].y1<e['row']['bbox'].y0-0.5]
        previous_bottom=max((ev['row']['bbox'].y1 for ev in earlier),default=0.0)
        desired_top=e['row']['bbox'].y0-10.5
        stem_top=max(0.0, previous_bottom+0.8, desired_top)
        # If the section title is unusually close, its bbox can overlap the
        # first formula line. Keep just enough clearance from the title.
        if heading_just_above:
            stem_top=max(stem_top,previous_bottom+0.8)
        # Extend to any nearby detached formula glyph that belongs to the first
        # line, but never cross the previous semantic boundary.
        if shared_for_question is None:
            floating=[row for row in page_lines(e['page'])
                if max(previous_bottom+0.8,e['row']['bbox'].y0-14)<=row['bbox'].y0<e['row']['bbox'].y0-0.5
                and row['bbox'].x0>e['row']['bbox'].x0+55
                and not QUESTION_RE.match(row['text'])
                and not ANSWER_RE.search(row['text'])
                and not SOLUTION_RE.search(row['text'])
                and not OPTION_RE.match(row['text'])]
            if floating:
                stem_top=max(previous_bottom+0.8,min(row['bbox'].y0 for row in floating)-0.8)
        stem_begin=(e['page'],stem_top)
        stems=[]
        # If a question has a separate chart / geometry figure placed below its
        # text but before A/a), split the stem into text strip + figure strip.
        bottom_figure_rect=None
        last_stem_row=None
        if first_opt and e['page']==first_opt['page']:
            candidate_rows=[]
            for row in page_lines(e['page']):
                if row['bbox'].y0 < e['row']['bbox'].y0-0.5 or row['bbox'].y0 >= first_opt['row']['bbox'].y0-0.35:
                    continue
                if OPTION_RE.match(row['text']) or SOLUTION_RE.search(row['text']) or ANSWER_RE.search(row['text']):
                    continue
                if row['bbox'].x0 <= e['row']['bbox'].x0 + 90:
                    candidate_rows.append(row)
            if candidate_rows:
                last_stem_row=max(candidate_rows,key=lambda row:row['bbox'].y0)
                last_text_bottom=max(row['bbox'].y1 for row in candidate_rows)
                gap_top=last_text_bottom+1.5
                gap_bottom=first_opt['row']['bbox'].y0-2.0
                figure_between=False
                if gap_bottom-gap_top > 18:
                    for rect in [d.get('rect') for d in doc[e['page']].get_drawings()] + _image_blocks(doc[e['page']]):
                        if rect is None:
                            continue
                        rect=fitz.Rect(rect)
                        if rect.y1 < gap_top or rect.y0 > gap_bottom:
                            continue
                        if rect.width < 22 or rect.height < 12:
                            continue
                        if rect.width*rect.height < 900 and max(rect.width, rect.height) < 80:
                            continue
                        figure_between=True
                        break
                if figure_between and first_opt['row']['bbox'].y0 - last_text_bottom > 26:
                    bottom_figure_rect=(e['page'], [max(0, doc[e['page']].rect.width*.045), gap_top,
                        min(doc[e['page']].rect.width, doc[e['page']].rect.width*.955), gap_bottom])
                    stem_end=(e['page'], last_text_bottom+0.8)
        separated_figure_rects=[]
        if stem_graph_left is not None and stem_begin[0]==stem_end[0]:
            page=doc[stem_begin[0]]
            left_text_rows=[row for row in page_lines(stem_begin[0])
                if row['bbox'].y1 >= stem_begin[1]-0.4
                and row['bbox'].y0 <= stem_end[1]+0.8
                and row['bbox'].x0 < stem_graph_left-10.0
                and not OPTION_RE.match(row['text'])
                and not ANSWER_RE.search(row['text'])
                and not SOLUTION_RE.search(row['text'])]
            left_rects=_row_groups_vertical_regions(page,left_text_rows,stem_first_x0,stem_graph_left-3.0,
                top=stem_begin[1],bottom=stem_end[1]+0.8,gap=8.5,pad_y=1.0)
            if left_rects:
                for rect in left_rects:
                    stems.append({'page':stem_begin[0],'rect':rect,'slot':'stem_text'})
            else:
                left_rect=[stem_first_x0, stem_begin[1], stem_graph_left-3.0, max(0,stem_end[1]-0.4)]
                if left_rect[2]-left_rect[0] >= 20 and left_rect[3]-left_rect[1] >= 6:
                    stems.append({'page':stem_begin[0],'rect':left_rect, 'slot':'stem_text'})
        else:
            stems=strips(stem_begin,stem_end, first_x0=stem_first_x0)
            for s in stems:
                s['slot']='stem_text'
        if stem_graph_rect is not None:
            gp, grect = stem_graph_rect
            graph_top=grect[1]
            if grect[3]-graph_top >= 6:
                stems.append({'page':gp,'rect':[grect[0],graph_top,grect[2],grect[3]],'tight':True,'slot':'stem_figure'})
                separated_figure_rects.append((gp,[grect[0],graph_top,grect[2],grect[3]]))
        elif bottom_figure_rect is not None:
            gp, grect = bottom_figure_rect
            if grect[3]-grect[1] >= 20:
                stems.append({'page':gp,'rect':grect,'tight':True,'slot':'stem_figure'})
                separated_figure_rects.append((gp,grect))
        # Short-answer / essay figures are sometimes embedded images rather
        # than PDF drawings and may start slightly ABOVE the question header.
        # Assign such an image to the question whose vertical band it overlaps,
        # blank it from the text raster, then append it as stem_figure.
        if sec in (3,4) and not opts:
            page=doc[e['page']]
            prior=[z for z in events[:start] if z['page']==e['page'] and z['kind'] in ('section','question','answer','solution')]
            prev_bottom=max((z['row']['bbox'].y1 for z in prior),default=0.0)
            ans_y=(first_ans['row']['bbox'].y0 if first_ans and first_ans['page']==e['page'] else None)
            img_candidates=[]
            for rect in _image_blocks(page):
                if rect.width < 55 or rect.height < 35:
                    continue
                if rect.x0 < page.rect.width*.25:
                    continue
                if rect.y1 <= e['row']['bbox'].y0+6:
                    continue
                if rect.y0 < prev_bottom-8:
                    continue
                if ans_y is not None and rect.y1 > ans_y+10:
                    continue
                img_candidates.append(rect)
            if img_candidates:
                rect=max(img_candidates,key=lambda r:r.width*r.height)
                blank_rect=[rect.x0-1.5,rect.y0-1.5,rect.x1+1.5,rect.y1+1.5]
                for s in stems:
                    if s['page']!=e['page']:
                        continue
                    sr=fitz.Rect(s['rect']); br=fitz.Rect(blank_rect)
                    if not (sr & br).is_empty:
                        existing=s.get('blank')
                        if existing is None:s['blank']=[blank_rect]
                        elif isinstance(existing,list) and existing and isinstance(existing[0],(list,tuple,fitz.Rect)):existing.append(blank_rect)
                        else:s['blank']=[existing,blank_rect]
                stems.append({'page':e['page'],'rect':blank_rect,'tight':True,'slot':'stem_figure'})
        if shared_for_question:
            shared_index, shared_event, shared_target_index, shared_target = shared_for_question
            shared_first_x0=max(0.0, shared_event['row']['bbox'].x0-2.5)
            shared_boundary=(shared_target['page'], max(shared_event['row']['bbox'].y1+0.25, shared_target['row']['bbox'].y0-0.35))
            for marker in events[shared_index+1:shared_target_index]:
                if marker['kind'] in ('answer','solution','question','section'):
                    shared_boundary=(marker['page'], marker['row']['bbox'].y0-0.35)
                    break
            shared_regions=strips((shared_event['page'], max(0, shared_event['row']['bbox'].y0-0.2)), shared_boundary, first_x0=shared_first_x0)
            for s in shared_regions:
                s['slot']='stem_text'
            if shared_regions:
                stems=shared_regions+stems
        # V5.27.11: the surrounding HTML already renders "Câu n". Blank only
        # that printed prefix from the first visual stem strip.
        question_label=_question_label_box(e)
        if question_label:
            for stem_item in stems:
                if stem_item.get('page')!=e['page'] or stem_item.get('slot')!='stem_text':
                    continue
                rect=fitz.Rect(stem_item.get('rect') or [0,0,0,0])
                if rect.intersects(fitz.Rect(question_label)):
                    existing=stem_item.get('blank')
                    if existing is None:
                        stem_item['blank']=[question_label]
                    elif isinstance(existing,list) and existing and isinstance(existing[0],(list,tuple,fitz.Rect)):
                        existing.append(question_label)
                    else:
                        stem_item['blank']=[existing,question_label]
                    break
        if not stems:raise VisualReviewRequired(f'Câu {e["number"]}: vùng cắt đề bài rỗng.')
        opts_out=[]
        # Compare options by bounding-box centers, not raw baseline: baseline
        # shifts occur for indices, fractions, chemistry charges and geometry.
        # A row may contain 1, 2 or 4 columns. If boundaries are ambiguous,
        # refuse to publish rather than attach another option to the wrong key.
        def opt_row(a,b):
            if a['page']!=b['page'] or a.get('kind')!='option' or b.get('kind')!='option' or abs(a['x']-b['x'])<30:return False
            ra=a['row']['bbox'];rb=b['row']['bbox']
            # Font superscripts shift the extracted baselines by several pt.
            # Two different rows can have overlapping glyph boxes, so require
            # close starts as well as vertical overlap / center proximity.
            top_gap=abs(ra.y0-rb.y0)
            overlap=max(0,min(ra.y1,rb.y1)-max(ra.y0,rb.y0))
            return top_gap<10.0 or (top_gap<14.0 and overlap>=.35*min(ra.height,rb.height))
        for oi,o in enumerate(opts):
            page=doc[o['page']]
            same_row=[z for z in opts if z is not o and opt_row(o,z)]
            # Locate next option column only on the *same visual row*.
            right=[z for z in same_row if z['x']>o['x']+6]
            nearest=min(right,key=lambda z:z['x']) if right else None
            # End vertically at the next distinct row / answer / solution.
            # We intentionally do not require A,B,C,D baselines to be sorted.
            later_markers=[z for z in (opts+solutions+answers)
                if z is not o and (z['page']>o['page'] or
                (z['page']==o['page'] and z['row']['bbox'].y0>o['row']['bbox'].y0+7.5
                 and not opt_row(o,z)))]
            next_row=min(later_markers,key=lambda z:(z['page'],z['row']['bbox'].y0)) if later_markers else None
            endpos=(next_row['page'],next_row['row']['bbox'].y0-.5) if next_row else limit
            if first_explanation is not None:
                if endpos > first_explanation:
                    endpos=(first_explanation[0],first_explanation[1]-1.2)
            x0=max(page.rect.width*.045,(o['row']['bbox'].x1-4.0) if o['row']['text'].strip() in ('A','B','C','D','a)','b)','c)','d)') else (o['end_x']+.10))
            has_left_peer=any(z['x']<o['x']-6 for z in same_row)
            wrap_rows=[row for row in page_lines(o['page'])
                if row['bbox'].y0 > o['row']['bbox'].y1+0.45
                and row['bbox'].y1 < (next_row['row']['bbox'].y0-0.6 if next_row and next_row['page']==o['page'] else page.rect.height-24)
                and row['bbox'].x0 <= o['row']['bbox'].x0+22.0
                and not OPTION_RE.match(row['text'])
                and not ANSWER_RE.search(row['text'])
                and not SOLUTION_RE.search(row['text'])]
            wrap_continues = bool(wrap_rows) and not has_left_peer
            if wrap_continues and sec in (1,2) and not nearest:
                # Long single-column chemistry options often wrap onto the next
                # line starting near the left page margin. Keep the continuation
                # line by widening the crop all the way to the content margin
                # and blank the printed label separately.
                x0=max(page.rect.width*.045, 0.0)
            if nearest:
                # Keep almost the full horizontal run for this option but stop
                # before the next printed label/column.
                x1=min(page.rect.width*.955, nearest['x']-4.0)
            else:
                x1=page.rect.width*.955
            # Section II two-column layout: a)/b) occupy row 1 and c)/d)
            # row 2. These are independent columns, not four consecutive
            # full-page strips. Use the matching next marker in each column.
            two_col_tf=(sec==2 and len(opts)==4 and
                len({z['page'] for z in opts})==1 and
                abs(opts[0]['x']-opts[1]['x'])>95 and
                abs(opts[2]['x']-opts[3]['x'])>95 and
                abs(opts[0]['row']['bbox'].y0-opts[1]['row']['bbox'].y0)<6 and
                abs(opts[2]['row']['bbox'].y0-opts[3]['row']['bbox'].y0)<6)
            if two_col_tf:
                split_x=(opts[0]['x']+opts[1]['x'])/2.0
                # The left mathematical expression may extend well beyond
                # the midpoint of the two labels; stop at the RIGHT label,
                # not at the midpoint between labels.
                x1=min(x1,opts[1]['x']-5.0) if o['key'] in ('A','C') else page.rect.width*.955
                # Allow multiline maths above the printed option marker,
                # while keeping the solutions on the other side of its row.
                lower_top=min(opts[2]['row']['bbox'].y0,opts[3]['row']['bbox'].y0)
                if o['key'] in ('A','B'):
                    endpos=(o['page'],lower_top-8.0)
                else:
                    endpos=(o['page'],min(endpos[1],first_explanation[1]-2.0)
                            if first_explanation and first_explanation[0]==o['page'] else endpos[1])
            # Options must be complete. Start slightly ABOVE the marker row to
            # preserve fractions / superscripts, but not so high that the tail
            # of the stem is captured again.
            bare_label=o['row']['text'].strip() in (
                'A','B','C','D','A.','B.','C.','D.',
                'a)','b)','c)','d)','a.','b.','c.','d.')
            complex_headroom = circled_options or bare_label
            y0=max(0,o['row']['bbox'].y0-(16.0 if complex_headroom else 2.2))
            # The option label may be vertically centred inside a tall
            # fraction. Preserve its numerator, but never pull the last
            # line of the question into the FIRST option row.
            prev_line=previous_stem_line(o['page'], o['row']['bbox'].y0)
            if prev_line is not None:
                # The preceding stem line's bounding box may overlap the
                # first option label even though printed glyphs do not.
                # Exclude that entire line from the cropped option image.
                # Only enforce the stem cutoff on the FIRST answer row;
                # the later rows (C/D or vertically stacked a/b/c/d) have
                # their own preceding options and must not lose content.
                if (first_opt and (opt_row(o,first_opt) or o is first_opt)) or ((o['page'],o['row']['bbox'].y0)<=(first_opt['page'], first_opt['row']['bbox'].y0+10)):
                    y0=max(y0,prev_line['bbox'].y1+0.20)
            figure_overlap_blanks=[[grect[0]-1.0,grect[1]-1.0,grect[2]+1.0,grect[3]+1.0]
                                   for gp,grect in separated_figure_rects
                                   if gp==o['page'] and grect[3] <= o['row']['bbox'].y0+18.0 and grect[3] >= o['row']['bbox'].y0-40.0]
            figure_directly_above=bool(figure_overlap_blanks)
            if figure_directly_above and bare_label and len(o['row']['text'].strip())>2:
                y0=max(y0,o['row']['bbox'].y0-0.5)
            # Prevent an option on the next line (notably d)) from carrying
            # the tail of the previous c) statement in its screenshot.
            previous=[z for z in opts if z is not o and z['page']==o['page']
                      and z['row']['bbox'].y1<=o['row']['bbox'].y0
                      and abs(z['x']-o['x'])<30]
            if previous:
                y0=max(y0,max(z['row']['bbox'].y1 for z in previous)+.4)
            same_col_prev=[z for z in opts if z is not o and z['page']==o['page']
                           and abs(z['x']-o['x'])<35
                           and z['row']['bbox'].y0<o['row']['bbox'].y0-5]
            if not same_row and same_col_prev:
                upper=max(same_col_prev,key=lambda z:z['row']['bbox'].y0)
                c1=(upper['row']['bbox'].y0+upper['row']['bbox'].y1)/2.0
                c2=(o['row']['bbox'].y0+o['row']['bbox'].y1)/2.0
                y0=max(y0,(c1+c2)/2.0)
            if endpos[0]==o['page']:
                y1=min(page.rect.height,endpos[1])
                if next_row and next_row['page']==o['page']:
                    y1=min(y1,next_row['row']['bbox'].y0-.35)
            else:
                y1=page.rect.height-24
            if endpos[0]>o['page'] and next_row and next_row['kind'] in ('solution','answer'):
                # One-row A/B/C/D immediately before a PDF page break: the
                # following page starts with the SOLUTION, not four additional
                # option images. V5.25.7 stitched the solution page to EACH
                # option, mixing answers and other questions together.
                if len(opts)==4 and all(opt_row(o,z) for z in opts if z is not o):
                    row_bottom=max(z['row']['bbox'].y1 for z in opts)
                    y1=min(y1,row_bottom+13.0)
                    endpos=(o['page'],y1)
            # Two-column MCQ with stacked fractions, e.g. OTC1 Phần I
            # Câu 11: B's denominator reaches BELOW D's extracted text bbox
            # top, while D's numerator begins ABOVE the printed D label.
            # Use a common cut line for each column rather than subtracting
            # 16pt from every option label (which copies B into D).
            two_row_mcq=(sec==1 and len(opts)==4 and
                len({z['page'] for z in opts})==1 and
                opts[0]['x']+55 < opts[1]['x'] and
                abs(opts[0]['x']-opts[2]['x'])<35 and
                abs(opts[1]['x']-opts[3]['x'])<35 and
                min(opts[2]['row']['bbox'].y0,opts[3]['row']['bbox'].y0) >
                    max(opts[0]['row']['bbox'].y0,opts[1]['row']['bbox'].y0)+12)
            if two_row_mcq:
                upper=opts[0] if o['key'] in ('A','C') else opts[1]
                lower=opts[2] if o['key'] in ('A','C') else opts[3]
                # Keep the right option column away from a graph/diagram that
                # occupies the far-right side of the same question. Mirror the
                # left-column width instead of letting B/D run to page edge.
                col_span=max(70.0, opts[1]['x']-opts[0]['x']-8.0)
                if o['key'] in ('B','D'):
                    x1=min(x1, opts[1]['x']+col_span)
                gap=lower['row']['bbox'].y0-upper['row']['bbox'].y1
                # Ordinary one-line text choices do not need extra space below
                # the upper row. A padded seam copied the next option into
                # choices A/B (Physics Q10). Preserve special headroom only
                # when the PDF option label has almost no text: a sign that
                # the expression is split into separate equation glyph runs.
                simple_text_rows=all(len(z['row']['text'].strip())>12 for z in opts)
                fallback_seam=((upper['row']['bbox'].y1+lower['row']['bbox'].y0)/2.0
                      if simple_text_rows else
                      lower['row']['bbox'].y0+(1.0 if gap<6 else -1.0))
                # V5.27.9: stacked fractions may extend outside the extracted
                # text bbox. Find the actual blank raster line separately in
                # each answer column. This preserves A/B denominators and C/D
                # numerators without copying content from the neighbouring row.
                col_left=max(page.rect.width*.045, min(upper['x'],lower['x'])-3.0)
                if o['key'] in ('A','C'):
                    col_right=min(page.rect.width*.955, opts[1]['x']-5.0)
                else:
                    col_right=page.rect.width*.955
                # Search only around the inter-row gap. Searching the whole
                # fraction box can mistake the whitespace between numerator
                # and fraction bar for the row separator.
                search_top=max(0,upper['row']['bbox'].y1-0.8)
                search_bottom=min(page.rect.height,lower['row']['bbox'].y0+5.5)
                seam=_find_ink_seam(page,col_left,col_right,search_top,search_bottom,fallback_seam)
                # Allow a tiny overlap around a blank seam. Raster trimming
                # removes the whitespace, while the formula glyphs stay whole.
                if o['key'] in ('A','B'):
                    y1=min(page.rect.height,seam+0.7)
                else:
                    y0=max(0,seam-0.7)
            if two_col_tf:
                lower_top=min(opts[2]['row']['bbox'].y0,opts[3]['row']['bbox'].y0)
                if o['key'] in ('A','B'):
                    y0=max(0,o['row']['bbox'].y0-1.7)
                    y1=min(page.rect.height,lower_top-8.0)
                else:
                    # d)'s numerator starts several points ABOVE its label.
                    y0=max(0,lower_top-5.5)
                    if first_explanation and first_explanation[0]==o['page']:
                        y1=min(y1,first_explanation[1]-3.0)
            # Vertically stacked single-column choices can have the NEXT
            # choice's numerator/fraction bar above its printed label. When the
            # current option itself is not multiline, cut at the midpoint of
            # the two label centers instead of at next_label.y0. This prevents
            # option A from stealing the top of B (Math Part II Q1, Physics Q16).
            same_col_next=[z for z in opts if z is not o and z['page']==o['page']
                           and abs(z['x']-o['x'])<35
                           and z['row']['bbox'].y0>o['row']['bbox'].y0+5]
            if not same_row and not wrap_continues and same_col_next:
                lower=min(same_col_next,key=lambda z:z['row']['bbox'].y0)
                c1=(o['row']['bbox'].y0+o['row']['bbox'].y1)/2.0
                c2=(lower['row']['bbox'].y0+lower['row']['bbox'].y1)/2.0
                seam=(c1+c2)/2.0
                if sec==2 and len(o['row']['text'].strip())>18:
                    seam=min(seam,o['row']['bbox'].y1+1.2)
                y1=min(y1,seam)
            # Clip against a separated stem figure ONLY when that figure
            # actually overlaps this option vertically. A figure that lives
            # above the answer list must not shorten long text options below
            # it (Physics Part II Q1-Q4).
            if stem_graph_rect is not None:
                gp,grect=stem_graph_rect
                if (o['page']==gp and x0 < grect[0]
                        and y1 > grect[1]+1.0 and y0 < grect[3]-1.0):
                    x1=min(x1,grect[0]-3.0)
            for gp,grect in separated_figure_rects:
                if o['page']!=gp:
                    continue
                if y0 < grect[3]-0.4 < y1 and grect[3] <= o['row']['bbox'].y0 + 14.0:
                    y0=max(y0,grect[3]+0.8)
            if x1-x0<8 or y1-y0<4:
                raise VisualReviewRequired(f'Câu {e["number"]}: không xác định được toàn bộ vùng phương án {o["key"]}.')
            region={'page':o['page'],'rect':[x0,y0,x1,y1]}
            if figure_overlap_blanks:
                region['blank']=list(figure_overlap_blanks)
            if wrap_continues and sec in (1,2) and not nearest and 'label_box' in o:
                if 'blank' in region:
                    region['blank'].append(o['label_box'])
                else:
                    region['blank']=[o['label_box']]
            regions=[region]
            # Cross-page options: preserve every original-page raster strip,
            # stitched vertically, without retyping any mathematical material.
            if endpos[0]>o['page']:
                for pn in range(o['page']+1,endpos[0]+1):
                    more=doc[pn]
                    bottom=(endpos[1] if pn==endpos[0] else more.rect.height-24)
                    if bottom>38:
                        regions.append({'page':pn,'rect':[more.rect.width*.045,35,
                                   more.rect.width*.955,bottom]})
            opts_out.append({'key':o['key'],
                             'content':_render_rects(doc,regions,media_dir,tight=True)})
        answer_val=None
        for item in answers:
            parsed=_parse_answer(item['row']['text'],sec)
            if sec==2 and parsed is None and 'Chọn' in item['row']['text']:
                # A multi-column answer key may be split across several PDF
                # text lines, but get_text('words') preserves its row geometry.
                words=sorted((w for w in doc[item['page']].get_text('words')
                       if abs(w[1]-item['y'])<6),key=lambda w:w[0])
                rowtext=' '.join(w[4] for w in words)
                pairs=re.findall(r'\b([abcd])\s+(đúng|sai)\b',rowtext,re.I)
                if len(pairs)==4 and [key.lower() for key,_ in pairs]==list('abcd'):
                    parsed=[value.lower()=='đúng' for _,value in pairs]
            if parsed is not None:answer_val=parsed;break
        if sec==1 and answer_val is None:
            for item in group:
                if item['kind']=='solution':
                    pos=group.index(item)
                    for row in group[pos+1:]:
                        m=re.search(r'Chọn\s+đáp\s+án\s*([ABCD])\b',row['row']['text'],re.I)
                        if m:answer_val=m.group(1).upper();break
                    break
        if sec in (1,2,3) and (answer_val is None or answer_val==[]):
            raise VisualReviewRequired(f'Phần {sec} câu {e["number"]}: chưa xác định được đáp án đúng từ PDF; tài liệu không cung cấp đáp án có thể xác minh; hãy bổ sung đáp án trong PDF nguồn trước khi tải lại.')
        sol_end=limit
        if first_ans and first_sol and (first_ans['page'],first_ans['y'])>(first_sol['page'],first_sol['y']):
            sol_end=(first_ans['page'],first_ans['y'])
        sol_data=strips((first_sol['page'],first_sol['y']),sol_end) if first_sol else []
        # If answer precedes solution: clip solution after the answer line, not
        # including its answer text. Used in files with an explicit answer line.
        if first_sol and first_ans and (first_ans['page'],first_ans['y'])<(first_sol['page'],first_sol['y']):
            pass
        counters[sec]+=1
        result.append({'original_order':len(result)+1,'section_order':counters[sec],
           'section':sec,'qtype':{1:'mcq',2:'tf',3:'short',4:'essay'}[sec],
           'stem':_render_rects(doc,stems,media_dir),
           'options':opts_out,'solution':_render_rects(doc,sol_data,media_dir),
           'correct_option':answer_val if sec==1 and isinstance(answer_val,str) else None,
           'tf_answers':answer_val if sec==2 and isinstance(answer_val,list) else [],
           'short_answer':answer_val if sec==3 and isinstance(answer_val,str) else None})
    return result


def _dominant_pdf_text_size(doc):
    """Weighted dominant body font size, used only for display scaling of exact crops.

    We never retype/OCR PDF content. If the source body is smaller than the
    site's 13.5pt exam standard, enlarge the raster proportionally so the
    original glyphs/math remain exact while the on-screen size matches Word/TeX.
    """
    weights={}
    try:
        for page in doc:
            data=page.get_text("dict")
            for block in data.get("blocks",[]):
                for line in block.get("lines",[]):
                    for span in line.get("spans",[]):
                        text=str(span.get("text") or "").strip()
                        size=float(span.get("size") or 0)
                        if not text or size < 8 or size > 18:
                            continue
                        key=round(size*2)/2.0
                        weights[key]=weights.get(key,0)+max(1,len(text))
        return max(weights,key=weights.get) if weights else PDF_VISUAL_TARGET_PT
    except Exception:
        return PDF_VISUAL_TARGET_PT

def extract_visual_questions(path,media_dir,manifest=None):
    """Return V5.25.3 compatible question dictionaries, with raster image blocks.

    `manifest` is the optional teacher-reviewed JSON region map for complicated
    diagrams/scans. No OCR, LaTeX conversion, or automatic answer guessing.
    """
    token=_CREATED.set([])
    try:
        return _extract_visual_questions_inner(path,media_dir,manifest)
    except Exception:
        for file in _CREATED.get():
            Path(file).unlink(missing_ok=True)
        raise
    finally:
        _CREATED.reset(token)

def _render_image_native(source,regions,media_dir):
    """Crop an image at ORIGINAL pixel coordinates: no PDF round-trip or resampling."""
    media_dir=Path(media_dir);media_dir.mkdir(parents=True,exist_ok=True)
    files=[];blocks=[]
    for region in regions:
        if int(region.get('page',0))!=0:
            raise VisualReviewRequired('Ảnh gốc có một trang; page phải là 0.')
        box=region.get('rect') or []
        if len(box)!=4:raise VisualReviewRequired('Vùng cắt ảnh cần đủ bốn tọa độ pixel.')
        l,t,r,b=map(int,box)
        if l<0 or t<0 or r>source.width or b>source.height or r<=l or b<=t:
            raise VisualReviewRequired('Vùng cắt nằm ngoài ảnh gốc.')
        with Image.open(source.filename) as original:
            image=original.convert('RGB').crop((l,t,r,b))
        if region.get('blank'):
            from PIL import ImageDraw
            bl=region['blank'];ImageDraw.Draw(image).rectangle(
              [bl[0]-l,bl[1]-t,bl[2]-l,bl[3]-t],fill='white')
        name='pdf_visual_'+uuid.uuid4().hex+'.png';dest=media_dir/name
        image.save(dest,optimize=True)
        created=_CREATED.get()
        if created is not None:created.append(dest)
        files.append(dest)
        display_width=min(PDF_VISUAL_MAX_CSS_WIDTH,round(image.width*96/DEFAULT_DPI))
        blocks.append({'type':'image','role':'figure','visual_pdf':True,
            'url':'/question-media/'+name,'display_width_px':display_width,
            'display_height_px':round(image.height*display_width/image.width),
            'word_width_cm':min(16,max(1.0,image.width*2.54/DEFAULT_DPI))})
    if len(files)>1:
        _merge_parts(files,media_dir)
        blocks=blocks[:1]
        with Image.open(files[0]) as img:
            blocks[0]['display_height_px']=round(img.height*blocks[0]['display_width_px']/img.width)
    return blocks


def _extract_visual_questions_inner(path,media_dir,manifest):
    path=Path(path); media_dir=Path(media_dir)
    if path.suffix.lower() in ('.png','.jpg','.jpeg','.webp'):
        if manifest is None:
            raise VisualReviewRequired('Ảnh đơn lẻ không có lớp văn bản và đáp án máy đọc được: không thể bảo đảm tự động chia câu và chấm điểm đúng. Vui lòng dùng PDF có lớp văn bản hoặc bản DOCX gốc.')
        # Crop native pixels directly. For source images manifest rectangles
        # are image pixel coordinates, not PDF point coordinates.
        with Image.open(path) as source:
            return _manual_questions(None,media_dir,manifest,
                render=lambda regions:_render_image_native(source,regions,media_dir))
    if path.suffix.lower()!='.pdf':raise VisualReviewRequired('Chỉ hỗ trợ PDF, PNG, JPG, JPEG, WEBP.')
    with fitz.open(path) as doc:
        if len(doc)>MAX_PAGES:raise VisualReviewRequired('PDF vượt giới hạn số trang an toàn.')
        body_pt=_dominant_pdf_text_size(doc)
        # Never shrink a source. Enlarge small-body PDFs only, capped to avoid
        # oversized cards from unusual metadata/font spans.
        display_scale=max(1.0,min(1.22,PDF_VISUAL_TARGET_PT/max(1.0,body_pt)))
        token=_DISPLAY_SCALE.set(display_scale)
        try:
            if manifest is not None:return _manual_questions(doc,media_dir,manifest)
            return _automatic_questions(doc,media_dir)
        finally:
            _DISPLAY_SCALE.reset(token)
