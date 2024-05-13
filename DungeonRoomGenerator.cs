using Godot;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

public partial class DungeonRoomGenerator : Node2D
{
	[Export] public TileMap tileMap;
	[Export] public int seed = 12345; // Default seed value
	[Export] public int width = 160; // Width of the room
	[Export] public int height = 80; // Height of the room
	[Export] public int initialFillPercent = 45; // Initial fill percentage
	[Export] public int smoothIterations = 5; // Number of cellular automata iterations
	[Export] public int tileSize = 16; // Size of each tile
	[Export] public Vector2I walkableTileAtlasCoords = new Vector2I(2, 2);
	[Export] public Vector2I obstacleTileAtlasCoords = new Vector2I(8, 7);
	[Export] public Vector2I playerAtlasCoords = new Vector2I(8, 3);
	[Export] public Vector2I goalAtlasCoords = new Vector2I(8, 3);
	[Export] public int playerStartPadding = 3; // Padding from the edges for player start position
	[Export] public int goalEndPadding = 3; // Padding from the edges for goal end position

	[Export] public int minDistanceBetweenPlayerAndGoal = 40; // Minimum distance between player and goal

	private Random _random;
	private bool[,] _roomShape;

	public override void _Ready()
	{
		tileMap = GetNode<TileMap>("TileMap");

		// Initialize random generator with seed
		_random = new Random(seed);

		// Start generation on a background thread
		GenerateRoomAsync();
	}

	private async void GenerateRoomAsync()
	{
		var stopwatch = Stopwatch.StartNew();

		// Run the room generation in a separate thread
		await Task.Run(() => GenerateRoom());

		stopwatch.Stop();
		GD.Print($"Room generated in: {stopwatch.ElapsedMilliseconds} ms");
	}



	private void GenerateRoom()
	{
		// Clear previous tiles
		tileMap.Clear();

		// Generate cave-like room shape
		// Find out how long it takes to generate the room, for that we will use a stopwatch
		var stopwatch = Stopwatch.StartNew();

		_roomShape = GenerateCaveShape(width, height, initialFillPercent, smoothIterations);

		stopwatch.Stop();
		Debug.WriteLine($"Room shape generated in: {stopwatch.ElapsedMilliseconds} ms");

		stopwatch.Restart();

		// Ensure all walkable areas are connected
		_roomShape = ConnectWalkableAreas(_roomShape);

		stopwatch.Stop();
		Debug.WriteLine($"Room connected in: {stopwatch.ElapsedMilliseconds} ms");
		stopwatch.Restart();


		// Fill the room shape with walkable tiles and obstacles
		FillRoomWithTiles(_roomShape);
		stopwatch.Stop();
		Debug.WriteLine($"Room filled in: {stopwatch.ElapsedMilliseconds} ms");

		stopwatch.Restart();
		// Place the player and goal
		PlacePlayerAndGoal(_roomShape);

		stopwatch.Stop();
		Debug.WriteLine($"Player and goal placed in: {stopwatch.ElapsedMilliseconds} ms");
	}

	private bool[,] GenerateCaveShape(int width, int height, int fillPercent, int iterations)
	{
		bool[,] shape = new bool[width, height];

		// Initialize with random fill
		Parallel.For(0, width, x =>
		{
			var localRandom = new Random(_random.Next());
			for (int y = 0; y < height; y++)
			{
				shape[x, y] = localRandom.Next(100) < fillPercent;
			}
		});

		// Apply cellular automata rules
		for (int i = 0; i < iterations; i++)
		{
			shape = SmoothShape(shape);
		}

		return shape;
	}

	private bool[,] SmoothShape(bool[,] shape)
	{
		int width = shape.GetLength(0);
		int height = shape.GetLength(1);
		bool[,] newShape = new bool[width, height];

		Parallel.For(0, width, x =>
		{
			for (int y = 0; y < height; y++)
			{
				int neighborWallCount = CountWallNeighbors(shape, x, y);

				if (neighborWallCount > 4)
				{
					newShape[x, y] = true;
				}
				else if (neighborWallCount < 4)
				{
					newShape[x, y] = false;
				}
				else
				{
					newShape[x, y] = shape[x, y];
				}
			}
		});

		return newShape;
	}

	private int CountWallNeighbors(bool[,] shape, int x, int y)
	{
		int width = shape.GetLength(0);
		int height = shape.GetLength(1);
		int wallCount = 0;

		for (int neighborX = x - 1; neighborX <= x + 1; neighborX++)
		{
			for (int neighborY = y - 1; neighborY <= y + 1; neighborY++)
			{
				if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height)
				{
					if (neighborX != x || neighborY != y)
					{
						wallCount += shape[neighborX, neighborY] ? 1 : 0;
					}
				}
				else
				{
					wallCount++;
				}
			}
		}

		return wallCount;
	}

	private bool[,] ConnectWalkableAreas(bool[,] roomShape)
	{
		int width = roomShape.GetLength(0);
		int height = roomShape.GetLength(1);

		bool[,] visited = new bool[width, height];
		HashSet<Vector2I> largestRegion = new HashSet<Vector2I>();

		// Find the largest connected region
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (!roomShape[x, y] && !visited[x, y])
				{
					HashSet<Vector2I> currentRegion = FloodFill(roomShape, visited, new Vector2I(x, y));
					if (currentRegion.Count > largestRegion.Count)
					{
						largestRegion = currentRegion;
					}
				}
			}
		}

		// Mark all other regions as walls
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (!largestRegion.Contains(new Vector2I(x, y)))
				{
					roomShape[x, y] = true;
				}
			}
		}

		return roomShape;
	}




	private HashSet<Vector2I> FloodFill(bool[,] roomShape, bool[,] visited, Vector2I start)
	{
		int width = roomShape.GetLength(0);
		int height = roomShape.GetLength(1);

		HashSet<Vector2I> region = new HashSet<Vector2I>();
		Queue<Vector2I> queue = new Queue<Vector2I>();
		queue.Enqueue(start);
		visited[start.X, start.Y] = true;

		Vector2I[] directions = new Vector2I[]
		{
		new Vector2I(1, 0),
		new Vector2I(-1, 0),
		new Vector2I(0, 1),
		new Vector2I(0, -1)
		};

		while (queue.Count > 0)
		{
			Vector2I current = queue.Dequeue();
			region.Add(current);

			foreach (Vector2I direction in directions)
			{
				Vector2I neighbor = current + direction;
				if (neighbor.X >= 0 && neighbor.X < width && neighbor.Y >= 0 && neighbor.Y < height)
				{
					if (!roomShape[neighbor.X, neighbor.Y] && !visited[neighbor.X, neighbor.Y])
					{
						visited[neighbor.X, neighbor.Y] = true;
						queue.Enqueue(neighbor);
					}
				}
			}
		}

		return region;
	}

	private void FillRoomWithTiles(bool[,] roomShape)
	{
		int width = roomShape.GetLength(0);
		int height = roomShape.GetLength(1);

		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (roomShape[x, y])
				{
					tileMap.SetCell(0, new Vector2I(x, y), 0, obstacleTileAtlasCoords, 0); // obstacle tile
				}
				else
				{
					tileMap.SetCell(0, new Vector2I(x, y), 0, walkableTileAtlasCoords, 0); // walkable tile
				}
			}
		}
	}

	private void PlacePlayerAndGoal(bool[,] roomShape)
	{
		int width = roomShape.GetLength(0);
		int height = roomShape.GetLength(1);

		// Find starting position concurrently with finding a random goal position far from the player
		Vector2I playerStartPosition = Vector2I.Zero;
		Vector2I goalEndPosition = Vector2I.Zero;

		Parallel.Invoke(
			() => playerStartPosition = FindWalkableTile(roomShape, new Vector2I(playerStartPadding, playerStartPadding), new Vector2I(width - playerStartPadding, height - playerStartPadding)),
			() => goalEndPosition = FindDistantWalkableTile(roomShape, playerStartPosition, minDistanceBetweenPlayerAndGoal)
		);

		// Move goal to ground if necessary
		goalEndPosition = MoveTileToGround(roomShape, goalEndPosition);

		// Generate a path between player and goal
		GeneratePathBetween(playerStartPosition, goalEndPosition);

		// Place the player and the goal
		tileMap.SetCell(0, playerStartPosition, 0, playerAtlasCoords, 0);
		tileMap.SetCell(0, goalEndPosition, 0, goalAtlasCoords, 0);
	}


	private Vector2I FindDistantWalkableTile(bool[,] roomShape, Vector2I start, int minDistance)
	{
		int width = roomShape.GetLength(0);
		int height = roomShape.GetLength(1);
		List<Vector2I> walkableTiles = new List<Vector2I>();

		// Collect all walkable tiles
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (!roomShape[x, y])
				{
					walkableTiles.Add(new Vector2I(x, y));
				}
			}
		}

		// Find a random tile that is at least `minDistance` away from `start`
		var distantTiles = walkableTiles.FindAll(tile => (tile - start).Length() >= minDistance);

		if (distantTiles.Count > 0)
		{
			return distantTiles[_random.Next(distantTiles.Count)];
		}
		else
		{
			// Fallback to any random walkable tile if no distant tiles found
			return walkableTiles[_random.Next(walkableTiles.Count)];
		}
	}


	private Vector2I FindWalkableTile(bool[,] roomShape, Vector2I rangeStart, Vector2I rangeEnd)
	{
		int width = roomShape.GetLength(0);
		int height = roomShape.GetLength(1);

		for (int x = Math.Min(rangeStart.X, rangeEnd.X); x <= Math.Max(rangeStart.X, rangeEnd.X); x++)
		{
			for (int y = Math.Min(rangeStart.Y, rangeEnd.Y); y <= Math.Max(rangeStart.Y, rangeEnd.Y); y++)
			{
				if (x >= 0 && y >= 0 && x < width && y < height && !roomShape[x, y])
				{
					return new Vector2I(x, y);
				}
			}
		}

		// Return a fallback position if no walkable tile is found within the specified range
		return new Vector2I(0, 0);
	}

	private Vector2I MoveTileToGround(bool[,] roomShape, Vector2I position)
	{
		int width = roomShape.GetLength(0);
		int height = roomShape.GetLength(1);
		int safety_limit = 1000;

		while (position.Y < height - 1 && !roomShape[position.X, position.Y + 1] && safety_limit-- > 0)
		{
			position.Y += 1;
		}

		return position;
	}
	private void GeneratePathBetween(Vector2I start, Vector2I end)
	{
		List<Vector2I> path = AStarPathfinding(start, end);

		// Ensure the path is cleared in the room
		foreach (Vector2I position in path)
		{
			tileMap.SetCell(0, position, 0, walkableTileAtlasCoords, 0);
		}
	}

	private List<Vector2I> AStarPathfinding(Vector2I start, Vector2I end)
	{
		HashSet<Vector2I> closedSet = new HashSet<Vector2I>();
		MinHeap<PathNode> openSet = new MinHeap<PathNode>();
		ConcurrentDictionary<Vector2I, Vector2I> cameFrom = new ConcurrentDictionary<Vector2I, Vector2I>();

		ConcurrentDictionary<Vector2I, int> gScore = new ConcurrentDictionary<Vector2I, int>
		{
			[start] = 0
		};

		ConcurrentDictionary<Vector2I, int> fScore = new ConcurrentDictionary<Vector2I, int>
		{
			[start] = HeuristicCostEstimate(start, end)
		};

		openSet.Add(new PathNode(start, fScore[start]));

		while (openSet.Count > 0)
		{
			PathNode currentNode = openSet.Pop();
			Vector2I current = currentNode.Position;

			if (current == end)
			{
				return ReconstructPath(cameFrom, current);
			}

			closedSet.Add(current);

			foreach (Vector2I neighbor in GetNeighbors(current))
			{
				if (closedSet.Contains(neighbor) || tileMap.GetCellAtlasCoords(0, neighbor, false) == obstacleTileAtlasCoords)
				{
					continue;
				}

				int tentativeGScore = gScore.TryGetValue(current, out int currentGScore) ? currentGScore + 1 : int.MaxValue;

				if (!gScore.ContainsKey(neighbor) || tentativeGScore < gScore[neighbor])
				{
					cameFrom[neighbor] = current;
					gScore[neighbor] = tentativeGScore;
					fScore[neighbor] = tentativeGScore + HeuristicCostEstimate(neighbor, end);

					PathNode neighborNode = new PathNode(neighbor, fScore[neighbor]);

					if (!openSet.Contains(neighborNode))
					{
						openSet.Add(neighborNode);
					}
				}
			}
		}

		return new List<Vector2I>(); // Return an empty path if no path is found
	}

	private class MinHeap<T> where T : IComparable<T>
	{
		private List<T> heap = new List<T>();

		public int Count => heap.Count;

		public void Add(T item)
		{
			heap.Add(item);
			HeapifyUp(heap.Count - 1);
		}

		public T Pop()
		{
			if (heap.Count == 0)
			{
				throw new InvalidOperationException("Heap is empty");
			}

			T root = heap[0];
			heap[0] = heap[heap.Count - 1];
			heap.RemoveAt(heap.Count - 1);
			HeapifyDown(0);

			return root;
		}

		public bool Contains(T item)
		{
			return heap.Contains(item);
		}

		private void HeapifyUp(int index)
		{
			int parent = (index - 1) / 2;
			if (index > 0 && heap[index].CompareTo(heap[parent]) < 0)
			{
				Swap(index, parent);
				HeapifyUp(parent);
			}
		}

		private void HeapifyDown(int index)
		{
			int left = 2 * index + 1;
			int right = 2 * index + 2;
			int smallest = index;

			if (left < heap.Count && heap[left].CompareTo(heap[smallest]) < 0)
			{
				smallest = left;
			}

			if (right < heap.Count && heap[right].CompareTo(heap[smallest]) < 0)
			{
				smallest = right;
			}

			if (smallest != index)
			{
				Swap(index, smallest);
				HeapifyDown(smallest);
			}
		}

		private void Swap(int indexA, int indexB)
		{
			T temp = heap[indexA];
			heap[indexA] = heap[indexB];
			heap[indexB] = temp;
		}
	}
	private int HeuristicCostEstimate(Vector2I a, Vector2I b)
	{
		// Using Manhattan distance as the heuristic
		return Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y);
	}

	private List<Vector2I> ReconstructPath(ConcurrentDictionary<Vector2I, Vector2I> cameFrom, Vector2I current)
	{
		List<Vector2I> path = new List<Vector2I> { current };

		while (cameFrom.TryGetValue(current, out Vector2I previous))
		{
			current = previous;
			path.Add(current);
		}

		path.Reverse();
		return path;
	}

	private IEnumerable<Vector2I> GetNeighbors(Vector2I position)
	{
		Vector2I[] directions = new Vector2I[]
		{
			new Vector2I(1, 0),
			new Vector2I(-1, 0),
			new Vector2I(0, 1),
			new Vector2I(0, -1)
		};

		foreach (Vector2I direction in directions)
		{
			Vector2I neighbor = position + direction;
			if (neighbor.X >= 0 && neighbor.X < width && neighbor.Y >= 0 && neighbor.Y < height)
			{
				yield return neighbor;
			}
		}
	}

	private class PathNode : IComparable<PathNode>
	{
		public Vector2I Position { get; }
		public int FScore { get; }

		public PathNode(Vector2I position, int fScore)
		{
			Position = position;
			FScore = fScore;
		}

		public int CompareTo(PathNode other)
		{
			return FScore.CompareTo(other.FScore);
		}
	}

	private class PriorityQueue<T> where T : IComparable<T>
	{
		private List<T> _elements = new List<T>();

		public int Count => _elements.Count;

		public void Enqueue(T item)
		{
			_elements.Add(item);
			_elements.Sort();
		}

		public T Dequeue()
		{
			T item = _elements[0];
			_elements.RemoveAt(0);
			return item;
		}

		public bool Contains(T item)
		{
			return _elements.Contains(item);
		}
	}

	public override void _Input(InputEvent @event)
	{
		if (@event.IsActionPressed("ui_accept"))
		{
			seed = _random.Next();
			GenerateRoom();
		}
	}
}
