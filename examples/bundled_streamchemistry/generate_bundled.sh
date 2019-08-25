for i in 6 7 8 9
do
    # Level zero
    python generate_bundled.py 0 $i
    # Level one
    python generate_bundled.py 1 $i
done
