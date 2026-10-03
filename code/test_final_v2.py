import random
import unittest
import numpy as np
from produce_final_v2 import scale_model, sample_size, placement, estimate_capacity, rgb_to_lab


class BoundaryTests(unittest.TestCase):
    def test_small_gt_uses_prior_and_truncates(self):
        prior=dict(q05=.01,q95=.3,cv=.2)
        m=scale_model([(0,0,40,40)],400,400,prior)
        self.assertEqual(m['variance_source'],'subset_cv')
        rng=random.Random(42); samples=[sample_size(m,rng) for _ in range(2000)]
        self.assertTrue(all(m['lower']<=s<=m['upper'] for s in samples))
        self.assertLess(abs(np.mean(samples)-40),1)
        self.assertIsNone(sample_size(dict(lower=20,upper=10),rng))

    def test_capacity_and_placement_respect_full_occupancy(self):
        im=np.full((240,240,3),150,np.uint8); lab=rgb_to_lab(im); ref=lab[0,0]
        hosts=[(95,95,125,125)]
        self.assertIsNone(placement(im,lab,ref,[(0,0,240,240)],hosts,(30,30),random.Random(7)))
        t=[dict(w=30,h=30,bw=25,bh=25,pixels=400,linear=25)]
        m=dict(mean=25,std=2,lower=22,upper=28)
        n,boxes=estimate_capacity(im,lab,ref,dict(boxes=[(0,0,240,240)],plane_boxes=hosts),m,t,random.Random(7))
        self.assertEqual(n,0); self.assertEqual(boxes,[])
        n,boxes=estimate_capacity(im,lab,ref,dict(boxes=hosts,plane_boxes=hosts),m,t,random.Random(7))
        self.assertGreater(n,0); self.assertLessEqual(n,6)
        self.assertEqual(len(boxes),n)


if __name__=='__main__': unittest.main()
