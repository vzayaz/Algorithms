import os
from unittest import TestCase

from algo.sorting import external_sort
from algo.sorting.external_sort import create_runs


class TestExternalSort(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestExternalSort, self).__init__(*args, **kwargs)

        self.data_dir = 'data/'


    def test_create_runs(self):

        create_runs(file_input= self.data_dir +'input1.txt', out_dir= self.data_dir, run_size=2, tmp_file_name="tmp_")
        self.assertTrue(os.path.isfile(self.data_dir + "tmp_0") and
                        os.path.isfile(self.data_dir + "tmp_1") and
                        os.path.isfile(self.data_dir + "tmp_2"))
        os.remove(self.data_dir + "tmp_0")
        os.remove(self.data_dir + "tmp_1")
        os.remove(self.data_dir + "tmp_2")



    def test_sort(self):


        external_sort.external_sort(self.data_dir + 'input1.txt', "sorted.txt", self.data_dir, 4, merge_by=2)
        self.assertTrue(os.path.isfile(self.data_dir + 'sorted.txt'))

