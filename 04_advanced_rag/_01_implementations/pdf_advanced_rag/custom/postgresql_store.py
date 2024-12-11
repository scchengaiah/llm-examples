import re
import uuid
from datetime import datetime
from typing import Iterator, List, Optional, Sequence, Tuple

import pytz
from langchain.storage.exceptions import InvalidKeyException
from langchain_core.stores import ByteStore
from langchain_postgres.vectorstores import _get_embedding_collection_store
from sqlalchemy import (
    UUID,
    Column,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    MetaData,
    Table,
    create_engine,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

CollectionStore = _get_embedding_collection_store()[1]

# Create a new metadata instance
metadata = MetaData()

# Define the byte store table with foreign key to collection table
# Using CollectionStore.__table__ to reference the existing table definition
langchain_byte_store = Table(
    "langchain_byte_store",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column(
        "collection_id",
        UUID(as_uuid=True),
        ForeignKey(CollectionStore.__table__.c.uuid, ondelete="CASCADE"),
        nullable=False,
    ),
    Column("value", LargeBinary, nullable=False),
    Column("created_at", DateTime, default=datetime.now(tz=pytz.utc)),
    Column(
        "updated_at",
        DateTime,
        default=datetime.now(tz=pytz.utc),
        onupdate=datetime.now(tz=pytz.utc),
    ),
    Index("idx_key_prefix", "id", postgresql_ops={"id": "desc"}),
)


class PostgresqlStore(ByteStore):
    """ByteStore implementation using PostgreSQL as backend.

    Examples:
        Create a PostgresqlStore instance and perform operations:

        .. code-block:: python
            store = PostgresqlStore(
                connection_string = "postgresql://user:password@host:port/database",
                collection = collection
            )

            # Set values
            store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values
            values = store.mget(["key1", "key2"])

            # Delete keys
            store.mdelete(["key1"])

            # Iterate over keys
            for key in store.yield_keys():
                print(key)
    """

    def __init__(
        self,
        connection_string: str,
        collection,
        **kwargs,
    ) -> None:
        """Initialize PostgreSQL store with connection parameters.

        Args:
            connection_string: str
                The connection string for the PostgreSQL database. It should be in the format:
                "postgresql+psycopg://user:password@host:port/database"
            collection: str
                The collection name to use in the PostgreSQL table.
            **kwargs: Additional connection parameters for SQLAlchemy
        """
        self.engine = create_engine(connection_string, **kwargs)
        self.collection = collection
        # Create the table if it doesn't exist
        langchain_byte_store.create(self.engine, checkfirst=True)
        self.Session = sessionmaker(bind=self.engine)

    def _validate_key(self, key: str) -> None:
        """Validate key format to prevent injection and invalid characters.

        Args:
            key: The key to validate

        Raises:
            InvalidKeyException: If key contains invalid characters
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            raise InvalidKeyException(f"Invalid characters in key: {key}")

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get values for the given keys.

        Args:
            keys: Sequence of keys to retrieve

        Returns:
            List of values (None for missing keys)
        """
        for key in keys:
            self._validate_key(key)

        try:
            with self.Session() as session:
                query = langchain_byte_store.select().where(
                    langchain_byte_store.c.id.in_(keys)
                )
                results = session.execute(query).fetchall()

                # Create a mapping of existing keys to values
                value_map = {str(result.id): result.value for result in results}

                # Return values in the same order as input keys
                return [value_map.get(key) for key in keys]
        except SQLAlchemyError as e:
            raise RuntimeError(f"Database error during mget: {str(e)}") from e

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set values for the given keys.

        Args:
            key_value_pairs: Sequence of (key, value) pairs to set
        """
        for key, _ in key_value_pairs:
            self._validate_key(key)

        try:
            with self.Session() as session:
                for key, value in key_value_pairs:
                    # Upsert implementation
                    stmt = (
                        langchain_byte_store.update()
                        .where(langchain_byte_store.c.id == key)
                        .values(value=value, updated_at=datetime.now(tz=pytz.utc))
                    )
                    result = session.execute(stmt)

                    if result.rowcount == 0:
                        stmt = langchain_byte_store.insert().values(
                            id=key, collection_id=self.collection.uuid, value=value
                        )
                        session.execute(stmt)

                session.commit()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Database error during mset: {str(e)}") from e

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the specified keys.

        Args:
            keys: Sequence of keys to delete
        """
        for key in keys:
            self._validate_key(key)

        try:
            with self.Session() as session:
                stmt = langchain_byte_store.delete().where(
                    langchain_byte_store.c.id.in_(keys)
                )
                session.execute(stmt)
                session.commit()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Database error during mdelete: {str(e)}") from e

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys matching the given prefix.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            Iterator of matching keys
        """
        try:
            with self.Session() as session:
                query = langchain_byte_store.select().with_only_columns(
                    [langchain_byte_store.c.id]
                )

                if prefix:
                    self._validate_key(prefix)
                    query = langchain_byte_store.select().with_only_columns(
                        [langchain_byte_store.c.id]
                    )

                # Yield keys in batches to handle large datasets
                batch_size = 1000
                offset = 0

                while True:
                    batch = session.execute(
                        query.order_by(langchain_byte_store.c.id)
                        .offset(offset)
                        .limit(batch_size)
                    ).fetchall()
                    if not batch:
                        break

                    for (id,) in batch:
                        yield str(id)

                    offset += batch_size
        except SQLAlchemyError as e:
            raise RuntimeError(f"Database error during yield_keys: {str(e)}") from e
