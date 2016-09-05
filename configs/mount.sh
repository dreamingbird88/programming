# List available disk/partitions
sudo fdisk -l

# Create a mount point (directory)
mkdir ~/Data

# Mount the device to that mount point
sudo mount -t ntfs /dev/sda2 ~/Data

# UNmount the device
# sudo umount ~/Data
