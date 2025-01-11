import json
import ZODB
import ZODB.blob
import transaction
from   ZODB.FileStorage import FileStorage
from   pathlib          import Path
from   os.path          import abspath, join
from   BTrees.OOBTree   import OOBTree


# GLOBAL VARIABLE
FDR = "filesystem_dir"
BDR = "blobs_dir"
FNM = "filesystem_name"
IMM = "in_memory"


class ZodbManager:
    def __init__(self):
        self.configuration = self.configuration_reading()
        self.database      = self.zodb_database()
        self.connection    = self.database.open()
        self.db_root       = self.connection.root()

    # CONFIGURATION READING
    @staticmethod
    def configuration_reading():
        conf_root_path = join(Path(abspath(__file__)).parents[1], "../Configurations", "Zodb")
        conf_file_path = join(conf_root_path, "zodb_conf.json")
        return json.load(open(conf_file_path))

    # DATABASE CONFIGURATION
    def zodb_database(self):
        # IF in_memory IS 0, THEN THE DATABASE WILL BE FILESYSTEM BASED
        if self.configuration[IMM] == 0:
            Path(self.configuration[FDR]).mkdir(parents=True, exist_ok=True)
            filesystem_full_path = join(self.configuration[FDR], self.configuration[FNM])
            storage              = FileStorage(file_name=str(filesystem_full_path), blob_dir=self.configuration[BDR])
            return ZODB.DB(storage, large_record_size = 1 << 30)
        # OTHERWISE, THE DATABASE WILL BE IN MEMORY
        return ZODB.DB(None, large_record_size = 1 << 30)

    # DATABASE COLLECTION CREATION
    def create_collection_ifnotexists(self, collection_name):
        if collection_name not in self.db_root:
            self.db_root[collection_name] = OOBTree()

    # ADD ELEMENT WITHIN A COLLECTION
    def add_element(self, collection_name, key, value):
        self.create_collection_ifnotexists(collection_name)
        self.db_root[collection_name].insert(key, value)
        self.commit()

    # GET SUBNETWORKS FROM DATABASE COLLECTION
    def get_subnetworks(self, collection_name, keys):
        graphs = [
            self.db_root[collection_name][key]
            for key in keys
            if self.db_root[collection_name].has_key(key)
        ]
        return graphs

    # TRANSACTION
    @staticmethod
    def commit(): transaction.commit()

    # CLOSE CONNECTION
    def close(self):
        self.connection.close()
        self.database.close()