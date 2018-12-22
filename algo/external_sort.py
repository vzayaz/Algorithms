"""
Implementation of external sort (rather naive, for educational purposes).
Problem statement:
Given a large input file of M lines and limited amount of RAM sort all lines in file.
"""
import os
import heapq

def create_runs(file_input, out_dir, run_size, tmp_file_name= "tmp_"):
    """
    Sort small chunks of big input file and output each chunk into a separate temp file
    :param file_input:
    :param out_dir:
    :param run_size:
    :return: number of runs
    """
    with open(file_input, "r") as finput:

        #  fill the array to sort:
        reached_end = False
        arr = []
        next_run_index = 0

        while not reached_end:
            arr.clear()
            for i in range(run_size):
                line = finput.readline()
                if line:
                    arr.append(int(line))
                else:
                    break
            if len(arr) == 0:
                # nothing to sort in this chunk
                reached_end=True
            else:
                text = "\n".join(map(str, sorted(arr)))
                with open(os.path.join(out_dir, tmp_file_name+str(next_run_index)), "w") as ftmp:
                    ftmp.write(text)
                next_run_index += 1

        return next_run_index

def merge_files(l_fnames, out_file, max_lines, remove=False):
    """
    Merges all the files in the l_fnames list and saves the result into out_file
    :param l_fnames:
    :param out_file:
    :param max_lines - the max allowed number of lines loaded into memory
    :return:
    """

    if max_lines < len(l_fnames):
        raise ValueError(f'Number of lines {max_lines} is less than number of files {len(l_fnames)}')

    try:
        l_files = [open(f, 'r') for f in l_fnames ]

        # a counter of lines currently loaded into memory (in heap really) per file
        l_lines_in_memory = [0] * len(l_files)
        heap = []

        n_files_left = len(l_files)

        with open(out_file, 'w') as fout:

                while n_files_left > 0:

                    max_lines_per_file = max_lines // n_files_left
                    # now for each file load the difference into the heap
                    for index, file in enumerate(l_files):
                        if file is None:
                            continue
                        diff = max_lines_per_file - l_lines_in_memory[index]

                        if diff > 0:
                            for counter in range(diff):
                                line = file.readline()
                                if not line:
                                    # file is empty! close it and set to None
                                    file.close()
                                    if remove:
                                        os.remove(l_fnames[index])
                                    l_files[index] = None
                                    n_files_left -= 1
                                    break
                                else:

                                    heapq.heappush(heap, (int(line), index))

# TODO: ok - now i see it can be optimiser - we should pop until each file has at least one element in memory
                    # each file has in memory at most max_lines_per_file elements. if less - the file was finished.
                    arr_result = []
                    for final_index in range(max_lines_per_file):
                        if not heap:
                            break
                        val, file_index = heapq.heappop(heap)
                        arr_result.append(val)
                        l_lines_in_memory[file_index] -= 1

                    content = '\n'.join(map(str, arr_result))
                    fout.write(content)
                    if n_files_left > 0 or len(heap) > 0:
                        fout.write('\n')

                arr_result.clear()
                while heap:
                    val, file_index = heapq.heappop(heap)
                    arr_result.append(val)

                fout.write('\n'.join(map(str, arr_result)))

    except Exception as ex:
        print("Something bad has happened")
        raise ex

    finally:
        for f in l_files:
            if f:
                f.close()


def external_sort(input_file, out_name, tmp_dir, max_lines_in_memory, merge_by=2):
    """
    Perform external sort of the lines in a large input file,
    loading into memory not mora than max_lines_in_memory lines
    :param input_file:
    :param tmp_dir:
    :param max_lines_in_memory:
    :param merge_by: max number of files to be merged in each merge attempt
    :return:
    """
    tmp_file_prefix = 'tmp_'
    n_runs = create_runs(input_file, tmp_dir, max_lines_in_memory, tmp_file_prefix)

    l_cur_files = [tmp_dir + tmp_file_prefix + str(i) for i in range(n_runs)]
    l_new_files = []

    generation = 0

    while len(l_cur_files) > 1:
        l_new_files.clear()
        index_in_gen = 0
        start = 0
        while start < len(l_cur_files):

            # read at most merge_by files:
            new_file = tmp_dir + tmp_file_prefix + str(generation) + '_' + str(index_in_gen)
            merge_files(l_cur_files[start:start+merge_by], new_file, max_lines_in_memory, remove=True)
            l_new_files.append(new_file)
            index_in_gen += 1
            start += merge_by

        l_cur_files, l_new_files = l_new_files, l_cur_files
        generation += 1

    assert len(l_cur_files) == 1, f"Only one file should be left in the end. Currently left: {l_cur_files}"

    os.rename(l_cur_files[0], os.path.join(os.path.dirname(input_file), out_name))

    return True

