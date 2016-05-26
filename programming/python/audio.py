"""
http://pydub.com/

git clone https://github.com/jiaaro/pydub.git

help:
/mnt/hgfs/Data/github/programming/python/audio.pydub.markdown

pip install pydub

sudo apt-add-repository ppa:jon-severinsson/ffmpeg
sudo apt-add-repository ppa:samrog131/ffmpeg

sudo apt-get install ffmpeg


"""

from pydub import AudioSegment
in_file = r"/mnt/hgfs/Data/Download/LuoJiShiWei/2015 (4).mp3"

out_file = r"/mnt/hgfs/Data/Download/LuoJiShiWei/test.mp3"
o2 = r"/mnt/hgfs/Data/Download/LuoJiShiWei/test2.mp3"

SplitMp3File(in_file)

def SplitMp3File(in_file, num_min=5, epsilon=0.5):
    """
    Read the mp3 file (in_file) and split it into pieces with the given minutes.
    Output files will be have suffice "_d" where d starts from 0.
    Args:
      in_file: the input mp3 file.
      num_min: the length (number of minutes) of each piece.
      epsilon: the last piece minimum length == num_min * epsilon.
    """
    song = AudioSegment.from_mp3(in_file)
    total = len(song)
    step = round(num_min * 60 * 1000)
    min_len = step * epsilon
    # Get the suffix length.
    suffix_len = 1
    t = total / step
    while t >= 10:
        suffix_len += 1
        t /= 10
    suffix_len = r'%' + str(suffix_len) + 'd'
    # Split file.
    start = 0
    end = step
    i = 0
    out_file = in_file[:-4] + r'-%s.mp3'
    while total - end > min_len:
        n = (suffix_len % i).replace(' ', '0')
        slip = song[start:end]
        slip.export(out_file % n, format="mp3")
        start = end
        end += step
        i += 1
    if start != 0 and total != end:
        n = (suffix_len % i).replace(' ', '0')
        slip = song[start:total]
        slip.export(out_file % n, format="mp3")
