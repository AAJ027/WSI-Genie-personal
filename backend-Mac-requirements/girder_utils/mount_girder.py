import multiprocessing
from girder_client import GirderClient
from girder_client_mount import mount_client
import time

class GirderMounter:
    def __init__(self):
        self.mount_process = None

    @staticmethod
    def _mount_girder(gc, mount_path):
        # gc = GirderClient()
        # gc.setToken(token=token)
        # Using foreground flag solves multiple issues
        #   - Killing the process unmounts the drive
        #   - We can pass token instead of username and password
        mount_client(path=mount_path, gc=gc, fuse_options="foreground", flatten=True)

    def mount(self, gc, path):
        if self.mount_process and self.mount_process.is_alive():
            print("Girder is already mounted")
            return

        self.mount_process = multiprocessing.Process(
            target=self._mount_girder,
            args=(gc, path)
        )
        self.mount_process.start()
        print(f"Girder mounted at {path}")

    def unmount(self):
        if self.mount_process and self.mount_process.is_alive():
            self.mount_process.kill()
            self.mount_process.join()
            print("Girder unmounted")
        else:
            print("Girder is not currently mounted")

# Test
if __name__ == "__main__":
    girder_mounter = GirderMounter()
    
    gc = GirderClient()
    gc.setToken("xmIGt8Gmq04pnl45pWFLRXXZ7xCPfMaqa9e8ejXRV0qwZdSgr1qeM6wE6vDqFyMl")
    # Mount
    girder_mounter.mount(
        gc = gc,
        path="/Users/parth/Desktop/digital_slide_archive/devops/dsa/mnt1"
    )
    
    time.sleep(10)
    girder_mounter.unmount()