import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .render import decode_selected,draw_hand,shape_projection,render_frame,SIZE


class RenderTests(unittest.TestCase):
    def test_relative_display_does_not_encode_global_position(self):
        points=np.arange(63,dtype=float).reshape(21,3)/100
        np.testing.assert_allclose(shape_projection(points),shape_projection(points+10),atol=1e-10)

    def test_gif_frame_selection_and_invalid_hand(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'clip.gif'
            images=[Image.new('RGB',(30,30),c) for c in ('red','green','blue')]
            images[0].save(path,save_all=True,append_images=images[1:],duration=100)
            picked=decode_selected(path,[0,2])
            self.assertEqual(picked[2].getpixel((0,0)),(0,0,255))
        im=Image.new('RGB',(30,30),'black');d=ImageDraw.Draw(im)
        draw_hand(d,np.full((21,2),np.nan),(0,0),1,'white')
        self.assertIsNone(im.getbbox())

    def test_missing_hand_keeps_original_visible(self):
        font=Path(os.environ.get('WINDIR',''))/'Fonts/malgun.ttf'
        if not font.is_file():
            try:font=Path(ImageFont.truetype('DejaVuSans.ttf',20).path)
            except OSError:self.skipTest('No Unicode font installed')
        row={'selected_index':None,'candidates':[]}
        meta={'clip_id':'test','domain':'unlinked_demo'}
        metric={'methods':{m:{'detected_frames':0,'target_frames':1} for m in ('raw','rotated')}}
        im=render_frame(Image.new('RGB',(640,360),'white'),{'frame':1,'roi_xywh':[100,100,120,120]},
                        {'raw':row,'rotated':row},meta,metric,font)
        self.assertEqual(im.size,SIZE)
        self.assertEqual(im.getpixel((816+200,176+200)),(255,255,255))


if __name__=='__main__':unittest.main()
