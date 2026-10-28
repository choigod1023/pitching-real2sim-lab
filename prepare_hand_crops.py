"""Raw-only inspection sheets; no projected points or outcome-based ROI fitting."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def run():
    root=Path(__file__).resolve().parent;lab=root.parent
    out=root/'artifacts/hand-image';out.mkdir(parents=True,exist_ok=True)
    crops={19:(700,290,950,570),20:(300,240,550,420),21:(440,380,720,700),22:(520,200,800,430)}
    for camera,box in crops.items():
        sheet=Image.new('RGB',(900,400),'#151a22');draw=ImageDraw.Draw(sheet)
        for j,frame in enumerate((974,977,979)):
            source=lab/('speed-journey/artifacts/release-observation' if camera in (20,21) else 'ball-coordinate-audit/artifacts/video-audit')/(f'camera{camera}_{frame}.png' if camera in (20,21) else f'camera{camera}_frame{frame}.png')
            im=Image.open(source).crop(box);d=ImageDraw.Draw(im)
            for x in range((box[0]//20+1)*20,box[2],20):
                d.line((x-box[0],0,x-box[0],im.height),fill='#4d5260');d.text((x-box[0]+1,2),str(x),fill='yellow')
            for y in range((box[1]//20+1)*20,box[3],20):
                d.line((0,y-box[1],im.width,y-box[1]),fill='#4d5260');d.text((1,y-box[1]),str(y),fill='yellow')
            sheet.paste(im,(j*300,40));draw.text((j*300+5,8),f'camera{camera} frame{frame} RAW ONLY',fill='white')
        sheet.save(out/f'raw-camera{camera}.png')


if __name__=='__main__':run()
